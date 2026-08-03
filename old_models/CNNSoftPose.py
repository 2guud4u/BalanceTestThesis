import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSoftPoseClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # CNN block: input [B, 2, 33, 5]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),  # [B, 16, 33, 5]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),                          # [B, 32, 33, 5]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                                          # [B, 32, 1, 1]
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),             # [B, 32]
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        x: Tensor of shape [B, 2*33*5]
        """
        x = x.view(-1, 2, 33, 5)  # Reshape for CNN
        features = self.cnn(x)    # [B, 32, 1, 1]
        if return_features:
            return features.view(features.size(0), -1)  # Return flattened CNN features
        return self.classifier(features)

    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(device)
