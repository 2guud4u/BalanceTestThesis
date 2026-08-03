import torch
import torch.nn as nn
import h5py
class SoftDino(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(64, num_classes)
        )


    def forward(self, x, return_features=False):
        if return_features:
            # Return features up to Linear(256->128) + batchnorm + relu, etc.
            # You can slice self.model layers if you want partial features, e.g.:
            # return self.model[:8](x)
            # Adjust indices based on how many layers you want to include.
            features = self.model[:8](x)
            return features
        return self.model(x)

    
    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(torch.device(device))
    