"""
Reusable network components.
"""

import torch
from torch import nn


class SumLayer(nn.Module):
    """
    Compute the sum along an axis.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Computes the sum along a dimension."""
        return torch.sum(x, dim=self.dim)


class ProdLayer(nn.Module):
    """
    Compute the product along an axis.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Computes the product along a dimension."""
        return torch.prod(x, dim=self.dim)


class MultiplyLayer(nn.Module):
    """
    Multiply with parameters.
    """

    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias


class SofplusTanhMul(nn.Module):
    """
    Softplus composed with tanh.
    """

    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

        self.prod_layer = ProdLayer(dim=-1)

    def forward(self, x):
        out = self.softplus(x)
        non_lin = self.tanh(out).detach()
        return self.prod_layer(torch.stack([x, non_lin], dim=-1))
