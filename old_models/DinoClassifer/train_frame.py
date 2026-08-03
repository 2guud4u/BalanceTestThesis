
# Create stratified train/val split
import os 
import sys
from network import FrameAwareDinoNetwork, FourLayerDinoNetwork, TwoLayerDinoNetwork, OneLayerDinoNetwork
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lossFun import FocalLoss
from dataloader import  get_dino_dataset, create_val_train_splits, get_label_mapping
from sklearn.model_selection import train_test_split
import torch
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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
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
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == val_y).sum().item()
                val_total += val_y.size(0)
        
        val_acc = 100 * val_correct / val_total
        # Step the scheduler
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
    
    features,labels, fps_list=get_dino_dataset(train_set_path, frame_aware=True, size=224)
    print(f"Features shape: {features[0].shape}, Labels shape: {labels[0].shape}")
    train_loader, val_loader, train_dataset, val_dataset, class_weights = create_val_train_splits(features,labels)
    # Update train_size for loss calculation

    train_size = len(train_dataset)
    print(val_dataset)
    print(f"Train size: {train_size}, Val size: {len(val_dataset)}")
    # Setup model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameAwareDinoNetwork().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=1.0, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, scheduler=scheduler)
    model.eval()
    truths = []
    preds = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            _, predicted = torch.max(outputs, 1)
            truths.extend(val_y.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    get_classifer_metrics(truths, preds, saveDir)
    print("Training and evaluation complete. Results saved to:", saveDir)
    print("Best model saved to:", best_model_path)
