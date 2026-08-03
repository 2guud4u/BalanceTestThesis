import torch
import torch.nn as nn
import h5py

class SinglePhaseBinaryDinoClassifier(nn.Module):
    def __init__(self, input_dim=1024, dropout_p=0.5):
        super(SinglePhaseBinaryDinoClassifier, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(64, 1)  # output logit
        )

    def forward(self, x):
        logits = self.model(x)
        return logits  
    
class PhaseDinoClassifier(nn.Module):
    def __init__(self, input_dim=1024, dropout_p=0.7):
        super(PhaseDinoClassifier, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(64, 4)  # output logit
        )

    def forward(self, x):
        logits = self.model(x)
        return logits  

