import torch.nn as nn


class KeypointPredictionModel(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=22):
        super(KeypointPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
