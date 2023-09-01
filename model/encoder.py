import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        z = self.encoder(x)
        y = F.normalize(self.head(z), dim=-1)
        return y
