
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half_dim = dim // 2

    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(
        torch.arange(half_dim, device=t.device) * -embeddings
    )

    embeddings = t[:, None] * embeddings[None, :]

    embeddings = torch.cat(
        [torch.sin(embeddings), torch.cos(embeddings)],
        dim=-1
    )

    return embeddings


class DDPMNetwork(nn.Module):
    def __init__(self, time_dim=128):
        super(DDPMNetwork, self).__init__()

        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)

        self.conv2 = nn.Conv2d(
            64, 128, 4, stride=2, padding=1
        )

        # Bottleneck
        self.bottleneck = nn.Conv2d(
            128, 128, 3, padding=1
        )

        # Decoder
        self.up = nn.ConvTranspose2d(
            128, 64, 4, stride=2, padding=1
        )

        self.output = nn.Conv2d(
            128, 1, 3, padding=1
        )

        self.time_projection = nn.Linear(
            time_dim, 128
        )

    def forward(self, x, t):
        time_embedding = sinusoidal_embedding(
            t, self.time_dim
        )

        time_embedding = self.time_mlp(time_embedding)

        time_embedding = self.time_projection(
            time_embedding
        )

        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        # Bottleneck + Time Embedding
        time_embedding = time_embedding[:, :, None, None]
        x2 = x2 + time_embedding
        x2 = F.relu(self.bottleneck(x2))

        # Decoder
        x3 = F.relu(self.up(x2))

        # Skip Connection
        x3 = torch.cat([x3, x1], dim=1)

        return self.output(x3)
