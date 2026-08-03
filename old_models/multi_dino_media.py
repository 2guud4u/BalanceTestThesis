import torch
import torch.nn as nn
import h5py


class Dino_Mediapipe_Network(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.Dino_Network = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.Mediapipe_Network = nn.Sequential(
            nn.Linear(2*33*5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64 + 64, num_classes)


    def forward(self, x, return_features=False):
        dino_features = self.Dino_Network(x[0])
        mp_features = self.Mediapipe_Network(x[1])
        
        combined_features = torch.cat((dino_features, mp_features), dim=1)
        
        if return_features:
            return combined_features
        
        return self.classifier(combined_features)

    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(torch.device(device))