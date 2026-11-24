
import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )

        # Mean and log variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        return z

    def forward(self, x):
        x = x.view(x.size(0), -1)

        hidden = self.encoder(x)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(-1, 1, 28, 28)

        return reconstructed, mu, logvar
