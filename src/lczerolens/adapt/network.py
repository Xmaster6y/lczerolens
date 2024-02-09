"""Reusable network components.

Classes
-------
SumLayer
    Compute the sum along an axis.
ProdLayer
    Compute the product along an axis.
MultiplyLayer
    Multiply with parameters.
SofplusTanhMul
    Softplus composed with tanh.
ElementwiseMultiplyUniform
    Distribute the relevance 100% to the input
SoftmaxEpsilon
    Softmax with epsilon.
MatrixMultiplicationEpsilon
    Matrix multiplication with epsilon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def stabilize(tensor, epsilon=1e-6):
    return tensor.add_(epsilon)


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


class ElementwiseMultiplyUniform(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, input_a, input_b):
        return input_a * input_b

    @staticmethod
    def backward(ctx, *grad_outputs):
        # relevance = grad_outputs[0].mul_(0.5)
        relevance = grad_outputs[0] * 0.5

        return relevance, relevance


class SoftmaxEpsilon(Function):
    @staticmethod
    def forward(ctx, inputs, dim):
        outputs = F.softmax(inputs, dim=dim)
        ctx.save_for_backward(inputs, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, output = ctx.saved_tensors

        relevance = (
            grad_outputs[0].sub_(
                output.mul_(grad_outputs[0].sum(-1, keepdim=True))
            )
        ).mul_(inputs)

        return (relevance, None)


class MatrixMultiplicationEpsilon(Function):
    @staticmethod
    def forward(ctx, input_a, input_b):
        outputs = torch.matmul(input_a, input_b)
        ctx.save_for_backward(input_a, input_b, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        input_a, input_b, outputs = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        out_relevance = out_relevance.div_(stabilize(outputs.mul_(2)))

        relevance_a = torch.matmul(
            out_relevance, input_b.permute(0, 1, -1, -2)
        ).mul_(input_a)
        relevance_b = torch.matmul(
            input_a.permute(0, 1, -1, -2), out_relevance
        ).mul_(input_b)

        return (relevance_a, relevance_b)
