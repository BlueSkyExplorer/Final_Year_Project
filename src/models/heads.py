import torch.nn as nn


class MultiClassHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class OrdinalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)
