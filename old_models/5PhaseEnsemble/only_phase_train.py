import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lossFun import FocalLoss
from dataloader import  get_dino_dataset, get_dataset
# Assume SinglePhaseBinaryDinoClassifier is your model for 4-class classification
# and get_dino_dataset returns features, labels, fps_list
from network import PhaseDinoClassifier
import numpy as np


def train_4class(model, train_loader, val_loader, criterion, optimizer,
                 num_epochs=1000, patience=10, scheduler=None,
                 device=None, train_size=None, saveDir=None, best_model_path=None):
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
            outputs = model(batch_X)  # [batch_size, 4]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            predicted = torch.argmax(outputs, dim=1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        epoch_loss = running_loss / train_size
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)  # [batch_size, 4]
                predicted = torch.argmax(outputs, dim=1)
                val_correct += (predicted == val_y).sum().item()
                val_total += val_y.size(0)

        val_acc = 100 * val_correct / val_total

        if scheduler:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}%")
            break

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")
    with open(os.path.join(saveDir, "training.txt"), "w") as f:
        f.write(f"Training completed in {end - start:.2f} seconds\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Total epochs: {epoch + 1}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    saveDir = os.path.join(os.path.dirname(__file__), "results", args[0])
    os.makedirs(saveDir, exist_ok=True)
    best_model_path = os.path.join(saveDir, "best_model.pth")

    train_set_path = args[1]
    features, labels, fps_list=get_dataset(train_set_path, "dinov3_features",frame_aware=True)


    indices = list(range(len(features)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Concatenate features/labels for train and val
    X_train = torch.cat([features[i] for i in train_indices], dim=0)
    y_train = torch.cat([labels[i] for i in train_indices], dim=0)
    X_val = torch.cat([features[i] for i in val_indices], dim=0)
    y_val = torch.cat([labels[i] for i in val_indices], dim=0)

    # Remove unlabeled frames if needed (example: label==4)
    X_train = X_train[y_train != 4]
    y_train = y_train[y_train != 4]
    X_val = X_val[y_val != 4]
    y_val = y_val[y_val != 4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute class weights for balanced loss
    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=y_train.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Setup model
    model = PhaseDinoClassifier(input_dim=1024+1).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Optional: WeightedRandomSampler
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Feature shape: {features[0].shape}, Label shape: {labels[0].shape}")


    # Train
    train_4class(model, train_loader, val_loader, criterion, optimizer,
                 num_epochs=1000, patience=10, scheduler=scheduler,
                 device=device, train_size=len(train_dataset), saveDir=saveDir,
                 best_model_path=best_model_path)

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Evaluation
    truths = []
    preds = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            predicted = torch.argmax(outputs, dim=1)
            truths.extend(val_y.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(truths, preds, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(cm, display_labels=["phase0","phase1","phase2","phase3"])
    disp.plot(cmap='Blues')
    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join(saveDir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report
    with open(os.path.join(saveDir, "classification_report.txt"), 'w') as f:
        f.write(classification_report(truths, preds, labels=[0,1,2,3], target_names=["phase0","phase1","phase2","phase3"]))

    print("Training and evaluation complete. Results saved to:", saveDir)
