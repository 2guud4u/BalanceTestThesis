# Create stratified train/val split
import os 
import sys
from network import SinglePhaseBinaryDinoClassifier
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lossFun import FocalLoss
from dataloader import  get_dataset, create_val_train_splits, get_label_mapping
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import h5py
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from eval_helper import get_classifer_metrics
import pytorch_lightning as pl

import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# Training loop with early stopping
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, scheduler=None, device=None, train_size=None, saveDir=None, best_model_path=None):
    best_val_acc = 0.0
    patience_counter = 0
    start = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
            # Calculate training accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Fixed: use sigmoid for binary classification
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        epoch_loss = running_loss / train_size
        train_acc = 100 * train_correct / train_total
        
        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                outputs = outputs.squeeze(1)  # Added: ensure consistent dimensions
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Fixed: use sigmoid for binary classification
                val_correct += (predicted == val_y).sum().item()
                val_total += val_y.size(0)
        
        val_acc = 100 * val_correct / val_total
        # Step the scheduler
        if scheduler:
            scheduler.step(val_acc)
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
        
        # Check for early stopping
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
    
    features,labels, fps_list=get_dataset(train_set_path, "dinov3_features",frame_aware=True)
    indices = list(range(len(features)))

    # Split video indices for train and val
    train_indices, val_indices = train_test_split(
        indices, test_size=.2, random_state=42
    )

    # Select train videos
    X_train = torch.cat([features[i] for i in train_indices], dim=0)
    y_train = torch.cat([labels[i] for i in train_indices], dim=0)
    # turn 0 to 1 and everything else to 0
    y_train = (y_train == 4).long()  # Assuming 0 is the positive class for phase one
    # Convert to tensor
    y_train = y_train.to(torch.float32)
    # Select val videos
    X_val = torch.cat([features[i] for i in val_indices], dim=0)
    y_val = torch.cat([labels[i] for i in val_indices], dim=0)
    y_val = (y_val == 4).long()  # Assuming 0 is the positive class for phase one
    # Convert to tensor
    y_val = y_val.to(torch.float32)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Setup device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    sample_weights = [class_weights[int(label)] for label in y_train]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"Train videos: {len(train_indices)}, Val videos: {len(val_indices)}")
    print(f"Features shape: {features[0].shape}, Labels shape: {labels[0].shape}")
    # Update train_size for loss calculation
    train_size = len(train_dataset)
    print(val_dataset)
    print(f"Train size: {train_size}, Val size: {len(val_dataset)}")
    
    # Setup model, loss, optimizer
    model = SinglePhaseBinaryDinoClassifier(input_dim=1024+1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0]).to(device)  # Fixed: use pos_weight for binary classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Pass required parameters to train function
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, 
          scheduler=scheduler, device=device, train_size=train_size, saveDir=saveDir, best_model_path=best_model_path)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    truths = []
    preds = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            outputs = outputs.squeeze(1)  # Added: ensure consistent dimensions
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Fixed: use sigmoid for binary classification
            truths.extend(val_y.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(truths, preds, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NonPhase", "Phase"])
    disp.plot(cmap='Blues')
    disp.ax_.set_title('Confusion Matrix')

    # Save figure
    plt.savefig(saveDir+"confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    # Save classification report
    with open(saveDir+"classification_report.txt", 'w') as f:
        f.write(classification_report(truths, preds, labels=[1, 0],target_names=["NonPhase", "Phase"]))

    print("Training and evaluation complete. Results saved to:", saveDir)
    print("Best model saved to:", best_model_path)