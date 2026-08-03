import torch
import torch.nn as nn
import h5py
class MP_Camera_Classifier(nn.Module):
    def __init__(self, input_dim=2*33*5, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_features=False):
        if return_features:
            # manually split the model if you want features
            features = self.model[:4](x)  # up to the second ReLU
            return features
        return self.model(x)

    
    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(torch.device(device))
    
