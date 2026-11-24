
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(SparseAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        reconstructed = self.decoder(z)

        reconstructed = reconstructed.view(-1, 1, 28, 28)

        return reconstructed, z
