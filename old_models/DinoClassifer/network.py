import torch
import torch.nn as nn
import h5py
class FourLayerDinoNetwork(nn.Module):
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
    

class TwoLayerDinoNetwork(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
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

class OneLayerDinoNetwork(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
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
    

class FrameAwareDinoNetwork(nn.Module):
    def __init__(self, input_dim=1024+1, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(64, num_classes)
        )


    def forward(self, x):

        x  
        return self.model(x)

    
    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(torch.device(device))

class AttentiveProbe(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_classes=5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # scalar attention score
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        context = (attn_weights * x).sum(dim=1)       # (batch_size, input_dim)
        context = context.view(context.size(0), -1)   # ensure 2D
        return self.classifier(context)

