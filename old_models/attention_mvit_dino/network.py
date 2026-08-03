import torch
import torch.nn as nn
import torch.nn.functional as F

class Dino_Mvit_Network(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.7):
        super().__init__()
        self.Dino_Network = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

        )
        self.Mvit_Network = nn.Sequential(
            nn.Linear(1152, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

        )

        # Cross-attention projection layers
        self.query_proj = nn.Linear(64, 64)
        self.key_proj = nn.Linear(64, 64)
        self.value_proj = nn.Linear(64, 64)

        # Final classifier
        self.classifier = nn.Linear(64 + 64, num_classes)

    def forward(self, x, return_features=False):
        dino_features = self.Dino_Network(x[0])  # shape: [B, 64]
        mp_features = self.Mvit_Network(x[1])  # shape: [B, 64]

        # Cross-attention: Dino queries MP
        Q = self.query_proj(dino_features)  # [B, 64]
        K = self.key_proj(mp_features)      # [B, 64]
        V = self.value_proj(mp_features)    # [B, 64]

        # Scaled dot-product attention (1-head)
        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (64 ** 0.5)  # [B, 1, 1]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, 1]
        attn_output = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)  # [B, 64]

        combined_features = torch.cat((dino_features, attn_output), dim=1)  # [B, 128]

        if return_features:
            return combined_features

        return self.classifier(combined_features)

    def load_model(self, model_path):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {device}")
        self.to(device)
        return self