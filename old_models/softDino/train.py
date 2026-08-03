# Create stratified train/val split
import os 
import sys
from network import SoftDino
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lossFun import FocalLoss
from dataloader import get_soft_dino_dataset, create_val_train_splits, get_label_mapping
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
import torch.nn.functional as F

# Training loop with early stopping based on validation loss
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=10, scheduler=None, device=None, train_size=None, best_model_path=None, saveDir=None):
    best_val_loss = float('inf')  # Track best validation loss instead of accuracy
    patience_counter = 0
    start = time.time()
    
    # Track training history for plotting
    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []  # Track validation losses
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # For KLDivLoss with soft labels (probability distributions)
            # Convert outputs to log probabilities
            log_probs = F.log_softmax(outputs, dim=1)
            
            # batch_y is already a probability distribution, use it directly
            # Ensure it's normalized (should sum to 1)
            targets = batch_y / (batch_y.sum(dim=1, keepdim=True) + 1e-8)  # Add epsilon to avoid division by zero
            
            loss = criterion(log_probs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
            # Calculate training accuracy using argmax of both outputs and targets
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(batch_y, 1)  # Get hard labels from soft labels
            train_correct += (predicted == true_labels).sum().item()
            train_total += batch_y.size(0)
        
        epoch_loss = running_loss / train_size
        train_acc = 100 * train_correct / train_total
        
        # Evaluate on validation set (including validation loss)
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                
                # Calculate validation loss
                log_probs = F.log_softmax(outputs, dim=1)
                targets = val_y / (val_y.sum(dim=1, keepdim=True) + 1e-8)
                val_loss = criterion(log_probs, targets)
                val_running_loss += val_loss.item() * val_X.size(0)
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(val_y, 1)  # Get hard labels from soft labels
                val_correct += (predicted == true_labels).sum().item()
                val_total += val_y.size(0)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_losses.append(val_epoch_loss)
        
        # Step the scheduler based on validation loss
        if scheduler:
            scheduler.step(val_epoch_loss)
        
        # Early stopping logic based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            # Save best model
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% *** NEW BEST ***")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
            break
    
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")
    
    # Save training summary
    if saveDir:
        with open(os.path.join(saveDir, "training.txt"), "w") as f:
            f.write(f"Training completed in {end - start:.2f} seconds\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            f.write(f"Final validation accuracy: {val_acc:.2f}%\n")
            f.write(f"Final train accuracy: {train_acc:.2f}%\n")
            f.write(f"Total epochs: {epoch + 1}\n")
        
        # Plot training curves
        plot_training_curves(train_losses, train_accs, val_accs, val_losses, saveDir)
    
    return best_val_loss, val_acc

def plot_training_curves(train_losses, train_accs, val_accs, val_losses, save_dir):
    """Plot and save training curves including validation loss"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation loss
    ax2.plot(val_losses, label='Validation Loss', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot both losses together
    ax3.plot(train_losses, label='Training Loss', color='blue')
    ax3.plot(val_losses, label='Validation Loss', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training vs Validation Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Plot accuracy
    ax4.plot(train_accs, label='Training Accuracy', color='blue')
    ax4.plot(val_accs, label='Validation Accuracy', color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Training and Validation Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")

def evaluate_soft_model(model, val_loader, device):
    """Comprehensive evaluation including soft label metrics"""
    model.eval()
    truths = []
    preds = []
    soft_truths = []
    soft_preds = []
    
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            
            # Hard predictions
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(val_y, 1)
            
            # Soft predictions
            soft_pred = F.softmax(outputs, dim=1)
            
            truths.extend(true_labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            soft_truths.extend(val_y.cpu().numpy())
            soft_preds.extend(soft_pred.cpu().numpy())
    
    return truths, preds, soft_truths, soft_preds

if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        saveDir = os.path.dirname(__file__)+"/results/"+args[0]+"/"
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        best_model_path = saveDir + "best_model.pth"
        print("Saving checkpoint to", best_model_path)
    except:
        raise ValueError("Please specify a save directory as the first argument.")

    try:
        train_set_path = args[1]
        if not os.path.exists(train_set_path):
            raise FileNotFoundError(f"Train set file not found: {train_set_path}")
    except:
        raise ValueError("Please specify the path to the train set file as the second argument.")
    
    # Load data
    features, labels, fps_list = get_soft_dino_dataset(train_set_path)
    print(f"Features shape: {features[0].shape}, Labels shape: {labels[0].shape}")
    
    # Check for any invalid soft labels
    print(f"Sample label: {labels[0][0]}")
    print(f"Label sum: {labels[0][0].sum():.4f}")
    
    train_loader, val_loader, train_dataset, val_dataset, class_weights = create_val_train_splits(features, labels, is_weighted=False)
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Setup model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SoftDino().to(device)
    
    # Use KLDivLoss for soft labels
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Change scheduler to monitor validation loss (mode='min' for loss minimization)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Train the model
    best_val_loss, final_val_acc = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=1000, 
        patience=10, 
        scheduler=scheduler,
        device=device,
        train_size=train_size,
        best_model_path=best_model_path,
        saveDir=saveDir
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Comprehensive evaluation
    truths, preds, soft_truths, soft_preds = evaluate_soft_model(model, val_loader, device)
    
    # Save standard metrics
    get_classifer_metrics(truths, preds, saveDir)
    
    # Additional soft label analysis
    import numpy as np
    
    # Calculate KL divergence on validation set
    kl_divs = []
    for true_dist, pred_dist in zip(soft_truths, soft_preds):
        # Add small epsilon to avoid log(0)
        true_dist = true_dist + 1e-8
        pred_dist = pred_dist + 1e-8
        
        # Normalize
        true_dist = true_dist / true_dist.sum()
        pred_dist = pred_dist / pred_dist.sum()
        
        kl_div = np.sum(true_dist * np.log(true_dist / pred_dist))
        kl_divs.append(kl_div)
    
    avg_kl_div = np.mean(kl_divs)
    
    # Save additional metrics
    with open(os.path.join(saveDir, "soft_metrics.txt"), "w") as f:
        f.write(f"Average KL Divergence: {avg_kl_div:.6f}\n")
        f.write(f"Hard accuracy: {100 * np.mean(np.array(truths) == np.array(preds)):.2f}%\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.2f}%\n")
    
    print("Training and evaluation complete. Results saved to:", saveDir)
    print("Best model saved to:", best_model_path)
    print(f"Average KL Divergence: {avg_kl_div:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")