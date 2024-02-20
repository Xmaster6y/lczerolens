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
    return tensor + epsilon * ((-1) ** (tensor < 0))


class AddEpsilonFunction(Function):
    @staticmethod
    def forward(ctx, input_a, input_b):
        output = input_a + input_b
        ctx.save_for_backward(input_a, input_b, output)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        input_a, input_b, output = ctx.saved_tensors
        out_relevance = grad_output[0] / stabilize(output)
        return out_relevance * input_a, out_relevance * input_b


class AddEpsilon(torch.nn.Module):
    def forward(self, x, y):
        return AddEpsilonFunction.apply(x, y)


class MatMulEpsilonFunction(Function):
    @staticmethod
    def forward(ctx, input, param):
        output = torch.matmul(input, param)
        ctx.save_for_backward(input, param, output)

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        input, param, output = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        out_relevance = out_relevance / stabilize(output)
        relevance = (out_relevance @ param.T) * input
        return relevance, None


class MatMulEpsilon(torch.nn.Module):
    def forward(self, x, y):
        return MatMulEpsilonFunction.apply(x, y)


class BilinearMatMulEpsilonFunction(Function):
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


class BilinearMatMulEpsilon(torch.nn.Module):
    def forward(self, x, y):
        return BilinearMatMulEpsilonFunction.apply(x, y)


class ElementwiseMultiplyUniformFunction(Function):
    @staticmethod
    def forward(ctx, input_a, input_b):
        return input_a * input_b

    @staticmethod
    def backward(ctx, *grad_outputs):
        relevance = grad_outputs[0] * 0.5

        return relevance, relevance


class SoftmaxEpsilonFunction(Function):
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
