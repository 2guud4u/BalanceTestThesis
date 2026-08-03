import torch
import torch.nn as nn
class MVITNetwork(nn.Module):
    def __init__(self, input_dim=1152, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),


            nn.Linear(64, num_classes)
        )


    def forward(self, x):

        return self.model(x)

    
    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(torch.device(device))
    