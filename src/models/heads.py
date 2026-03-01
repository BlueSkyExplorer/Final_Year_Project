import torch.nn as nn


class MultiClassHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.fc = nn.Sequential(
                nn.Identity(),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x):
        return self.fc(x)


class OrdinalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes - 1),
            )
        else:
            self.fc = nn.Sequential(
                nn.Identity(),
                nn.Linear(in_features, num_classes - 1),
            )

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()
        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
            )
        else:
            self.fc = nn.Sequential(
                nn.Identity(),
                nn.Linear(in_features, 1),
            )

    def forward(self, x):
        return self.fc(x).squeeze(1)
