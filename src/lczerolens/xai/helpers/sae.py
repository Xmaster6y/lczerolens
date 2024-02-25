"""
Defines the dictionary classes
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class AutoEncoder(nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size, pre_bias=False):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.pre_bias = pre_bias

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.activation_dim,
                    self.dict_size,
                )
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))
        self.relu = nn.ReLU()
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.dict_size,
                    self.activation_dim,
                )
            )
        )
        self.normalize_decoder_()
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.activation_dim,
            )
        )

    @torch.no_grad()
    def normalize_decoder_(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    def encode(self, x):
        return x @ self.W_enc + self.b_enc

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well
            as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode"
            where features are masked
        """
        if self.pre_bias:
            x = x - self.b_dec
        f_pre = self.encode(x)
        out = TensorDict({}, batch_size=x.shape[0])
        if ghost_mask is not None:
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            x_ghost = self.decode(f_ghost)
            out["x_ghost"] = x_ghost
        f = self.relu(f_pre)
        if output_features:
            out["features"] = f
        x_hat = self.decode(f)
        out["x_hat"] = x_hat
        return out
