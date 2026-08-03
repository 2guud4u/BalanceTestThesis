# Create stratified train/val split
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataloader import   create_val_train_splits, get_label_mapping, get_camera_pose_dataset
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
from network import MP_Camera_Classifier



# Training loop with early stopping
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10):
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

    features,labels=get_camera_pose_dataset("/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt")

    train_loader, val_loader, train_dataset, val_dataset, class_weights = create_val_train_splits(features,labels)

    # Update train_size for loss calculation

    train_size = len(train_dataset)

    # Setup model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MP_Camera_Classifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10)
    get_classifer_metrics(model, val_loader, saveDir,device)
    print("Training and evaluation complete. Results saved to:", saveDir)
    print("Best model saved to:", best_model_path)
