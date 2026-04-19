import torch
import torch.nn as nn

class RiskLSTM(nn.Module):
    def __init__(self, input_size=3, hidden=32):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))
