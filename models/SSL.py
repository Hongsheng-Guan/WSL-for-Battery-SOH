import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,50),
        )
        self.regressor = nn.Sequential(nn.Linear(32,16),nn.Linear(16,1))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output= self.regressor(encoded)
        return decoded, output