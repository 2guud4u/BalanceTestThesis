
# Create stratified train/val split
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import h5py
# move dir to parent directory
import sys
import os
import numpy
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import train_test_split
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from network import Dino_Mvit_Network
from dataloader import  get_dino_dataset, create_val_train_splits, get_world_pose_dataset, get_mvit_dataset
from eval_helper import get_classifer_metrics
from lossFun import FocalLoss


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, scheduler=None):
    best_val_acc = 0.0
    patience_counter = 0
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = [x.to(device) for x in batch_X], batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X[0].size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        epoch_loss = running_loss / train_size
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = [x.to(device) for x in val_X], val_y.to(device)
                outputs = model(val_X)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == val_y).sum().item()
                val_total += val_y.size(0)
        
        val_acc = 100 * val_correct / val_total

        # Step the scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.2f}%")
            break
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")
    with open(os.path.join(saveDir, "training.txt"), "w") as f:
        f.write(f"Training completed in {end - start:.2f} seconds\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Best train accuracy: {train_acc:.2f}%\n")
        f.write(f"Total epochs: {epoch + 1}\n")

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (dino_feature, mp_feature, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dino_feat, mp_feat, label = self.data[idx]
        return (dino_feat, mp_feat), label

if __name__ == "__main__":

    args = sys.argv[1:]
    try:
        saveDir = os.path.dirname(__file__)+"/results/"+args[0]+"/"
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        best_model_path = saveDir + "best_model.pth"
        print("saving checkpoint to", best_model_path)
    except:
        raise ValueError("Please specify a save directory as the first argument.")

    try:
        train_set_path =  args[1]
        if not os.path.exists(train_set_path):
            raise FileNotFoundError(f"Train set file not found: {train_set_path}")
    except:
        raise ValueError("Please specify the path to the train set file as the second argument.")
    

    dino_features,labels, fps=get_dino_dataset(train_set_path)
    mvit_features,labels, fps_list=get_mvit_dataset(train_set_path)


    # Get indices for videos
    indices = list(range(len(dino_features)))

    # Split video indices for train and val
    train_indices, val_indices = train_test_split(
        indices, test_size=.2, random_state=42
    )
    # Combine features
    # dino_features = torch.cat(dino_features, dim=0)
    # mp_features = torch.cat(mp_features, dim=0)
    train_dino = torch.cat([dino_features[i] for i in train_indices], dim=0)
    train_mvit = torch.cat([mvit_features[i] for i in train_indices], dim=0)
    train_labels = torch.cat([labels[i] for i in train_indices], dim=0)


    combined_train_data = list(zip(train_dino, train_mvit, train_labels))


    val_dino = torch.cat([dino_features[i] for i in val_indices], dim=0)
    val_mvit = torch.cat([mvit_features[i] for i in val_indices], dim=0)
    val_labels = torch.cat([labels[i] for i in val_indices], dim=0)
    combined_val_data = list(zip(val_dino, val_mvit, val_labels))
    #class_weights
    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2,3,4]), y=train_labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda')

    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create Dataset objects
    train_dataset = MultiModalDataset(combined_train_data)
    val_dataset = MultiModalDataset(combined_val_data)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # train_loader, val_loader, train_dataset, val_dataset = create_val_train_splits(features,labels)

    # Update train_size for loss calculation

    train_size = len(train_dataset)

    # Setup model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dino_Mvit_Network().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Add scheduler here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    # Training loop with early stopping
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, scheduler=scheduler)
    #get validation metrics
    print("Evaluating model on validation set...")
    preds = []
    truths = []
    model.eval()
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = [x.to(device) for x in val_X], val_y.to(device)
            outputs = model(val_X)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(val_y.cpu().numpy())

    get_classifer_metrics(truths, preds, saveDir)
    print("Training and evaluation complete. Results saved to:", saveDir)
    print("Best model saved to:", best_model_path)