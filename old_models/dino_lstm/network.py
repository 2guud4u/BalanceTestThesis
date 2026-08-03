import torch
import torch.nn as nn
class Dino_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=1, dropout=0.5):
        
        super(Dino_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # Get the last time
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out
    