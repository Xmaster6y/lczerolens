"""Function classes to apply the LRP rules to the layers of the network.

Classes
-------
ElementwiseMultiplyUniform
    Distribute the relevance 100% to the input
SoftmaxEpsilon
    Softmax with epsilon.
MatrixMultiplicationEpsilon
    Matrix multiplication with epsilon.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function


def stabilize(tensor, epsilon=1e-6):
    return tensor + epsilon


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
            grad_outputs[0] - (output * grad_outputs[0].sum(-1, keepdim=True))
        ) * inputs

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

        out_relevance = out_relevance / stabilize(2 * outputs)

        relevance_a = torch.matmul(
            out_relevance, input_b.permute(0, 1, -1, -2)
        ).mul_(input_a)
        relevance_b = torch.matmul(
            input_a.permute(0, 1, -1, -2), out_relevance
        ).mul_(input_b)

        return (relevance_a, relevance_b)
