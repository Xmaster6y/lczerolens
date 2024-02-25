"""
Defines the dictionary classes
"""

import torch as t
import torch.nn as nn
from tensordict import TensorDict


class AutoEncoder(nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.relu = nn.ReLU()

    def encode(self, x):
        return self.relu(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well
            as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode"
            where features are masked
        """
        f_pre = self.encoder(x - self.bias)
        out = TensorDict({}, batch_size=x.shape[0])
        if ghost_mask is not None:
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            x_ghost = self.decoder(f_ghost)
            out["x_ghost"] = x_ghost
        f = self.relu(f_pre)
        if output_features:
            out["features"] = f
        x_hat = self.decode(f)
        out["x_hat"] = x_hat
        return out
