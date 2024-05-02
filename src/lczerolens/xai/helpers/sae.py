"""
Defines the dictionary classes
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class AutoEncoder(nn.Module):
    """
    A 3-layers autoencoder.
    """

    def __init__(self, activation_dim, dict_size, pre_bias=False, less_than_1=False):
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
        self.D = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.dict_size,
                    self.activation_dim,
                )
            )
        )
        self.normalize_dict_(less_than_1=less_than_1)
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.activation_dim,
                    self.activation_dim,
                )
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.activation_dim,
            )
        )

    @torch.no_grad()
    def normalize_dict_(self, less_than_1):
        D_norm = self.D.norm(dim=1)
        if less_than_1:
            greater_than_1_mask = D_norm > 1
            self.D[greater_than_1_mask] /= D_norm[greater_than_1_mask].unsqueeze(1)
        else:
            self.D /= D_norm.unsqueeze(1)

    def encode(self, x):
        return x @ self.W_enc + self.b_enc

    def decode(self, x_f):
        return x_f @ self.W_dec + self.b_dec

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
            x_ghost = self.decode(f_ghost @ self.D)
            out["x_ghost"] = x_ghost
        f = self.relu(f_pre)
        if output_features:
            out["features"] = f
        x_f = f @ self.D
        x_hat = self.decode(x_f)
        out["x_hat"] = x_hat
        return out
