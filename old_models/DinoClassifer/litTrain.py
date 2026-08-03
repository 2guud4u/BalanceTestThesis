# PyTorch Lightning version of the training script
import os 
import sys
from network import FourLayerDinoNetwork, TwoLayerDinoNetwork
from sklearn.utils.class_weight import compute_class_weight
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lossFun import FocalLoss
from dataloader import get_dino_dataset, create_val_train_splits, get_label_mapping
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F


class DinoLightningModule(pl.LightningModule):
    def __init__(self, model_class=TwoLayerDinoNetwork, class_weights=None, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model_class()
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Initialize loss function
        self.criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
        
        # For tracking predictions during validation
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Store predictions and targets for epoch-end calculations
        self.validation_step_outputs.append({
            'loss': loss,
            'acc': acc,
            'preds': preds,
            'targets': y
        })
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Aggregate all predictions and targets from the epoch
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Calculate epoch-level metrics
        epoch_acc = (all_preds == all_targets).float().mean()
        self.log('val_acc_epoch', epoch_acc, prog_bar=True)
        
        # Clear the list for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Configure scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, factor=0.5, verbose=True
            ),
            'monitor': 'val_acc',
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def evaluate_model(model, val_loader, save_dir, device):
    """Evaluate the model and save metrics"""
    model.eval()
    truths = []
    preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            truths.extend(y.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    
    get_classifer_metrics(truths, preds, save_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        saveDir = os.path.dirname(__file__)+"/results/"+args[0]+"/"
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        print("Saving results to:", saveDir)
    except:
        raise ValueError("Please specify a save directory as the first argument.")

    try:
        train_set_path = args[1]
        if not os.path.exists(train_set_path):
            raise FileNotFoundError(f"Train set file not found: {train_set_path}")
    except:
        raise ValueError("Please specify the path to the train set file as the second argument.")
    
    # Load and prepare data
    features, labels, fps_list = get_dino_dataset(train_set_path)
    print(f"Features shape: {features[0].shape}, Labels shape: {labels[0].shape}")
    train_loader, val_loader, train_dataset, val_dataset, class_weights = create_val_train_splits(features, labels)
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Initialize Lightning module
    model = DinoLightningModule(
        model_class=TwoLayerDinoNetwork,
        class_weights=class_weights,
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=10,
        mode='max',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=saveDir,
        filename='best_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=saveDir,
        name='dino_training',
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator='auto',  # Automatically detects GPU/CPU
        devices='auto',      # Automatically detects available devices
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()
    
    # Load the best model for evaluation and save as .pth
    best_model_path = checkpoint_callback.best_model_path
    best_pth_path = os.path.join(saveDir, "best_model.pth")
    
    if best_model_path:
        # Load the full Lightning checkpoint
        lightning_model = DinoLightningModule.load_from_checkpoint(best_model_path)
        
        # Extract just the model weights and save as .pth
        torch.save(lightning_model.model.state_dict(), best_pth_path)
        print(f"Loaded best model from: {best_model_path}")
        print(f"Saved model weights to: {best_pth_path}")
        
        # Use the loaded model for evaluation
        model = lightning_model
    else:
        # Save current model state as .pth
        torch.save(model.model.state_dict(), best_pth_path)
        print(f"Saved final model weights to: {best_pth_path}")
    # Evaluate on validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    evaluate_model(model, val_loader, saveDir, device)
    
    # Save training summary
    training_time = end_time - start_time
    best_val_acc = trainer.callback_metrics.get('val_acc', 0.0)
    
    with open(os.path.join(saveDir, "training_summary.txt"), "w") as f:
        f.write(f"Training completed in {training_time:.2f} seconds\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
        f.write(f"Best model saved to: {best_model_path}\n")
    
    print("Training and evaluation complete!")
    print(f"Results saved to: {saveDir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")