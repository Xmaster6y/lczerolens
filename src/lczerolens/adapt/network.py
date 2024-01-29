"""
Custom network architectures.
"""

import re

import torch
from tensordict import TensorDict
from torch import nn

from . import constants


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


class SeLayer(nn.Module):
    """
    Squeeze and excitation layer.
    """

    def __init__(self, n_hidden, n_hidden_red=32) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_hidden_red = n_hidden_red

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(n_hidden, n_hidden_red)
        self.linear2 = nn.Linear(n_hidden_red, n_hidden * 2)
        self.sum_layer = SumLayer(dim=-1)
        self.prod_layer = ProdLayer(dim=-1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.avg_pool(x)
        out = out.view(-1, self.n_hidden)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = out.view(-1, self.n_hidden * 2, 1, 1)

        out1, out2 = out.split(self.n_hidden, dim=1)
        non_lin = self.sigmoid(out1).detach()
        out1 = self.prod_layer(
            torch.stack([residual, non_lin.repeat(1, 1, 8, 8)], dim=-1)
        )
        return self.sum_layer(
            torch.stack([out1, out2.repeat(1, 1, 8, 8)], dim=-1)
        )


class SeBlock(nn.Module):
    """
    SE ResNet block.
    """

    def __init__(self, n_hidden, n_hidden_red=32) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_hidden_red = n_hidden_red

        self.conv1 = nn.Conv2d(n_hidden, n_hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, 3, padding=1)
        self.se_layer = SeLayer(n_hidden, n_hidden_red)
        self.sum_layer = SumLayer(dim=-1)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se_layer(out)
        out = self.sum_layer(torch.stack([out, residual], dim=-1))
        return self.relu(out)


class PolicyHead(nn.Module):
    """
    Policy head.
    """

    def __init__(self, n_hidden) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.conv1 = nn.Conv2d(n_hidden, n_hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(n_hidden, 80, 3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.view(-1, 80 * 64)
        out = out.gather(
            1,
            torch.tensor(constants.GATHER_INDICES)
            .unsqueeze(0)
            .repeat(out.shape[0], 1)
            .to(out.device),
        )
        return out


class ValueHead(nn.Module):
    """
    Value head.
    """

    def __init__(self, n_hidden) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.conv = nn.Conv2d(n_hidden, 32, 1)
        self.linear1 = nn.Linear(32 * 64, 128)
        self.linear2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 32 * 64)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.tanh(out)
        return out


class MlhHead(nn.Module):
    """
    MLH head.
    """

    def __init__(self, n_hidden) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.conv = nn.Conv2d(n_hidden, 8, 1)
        self.linear1 = nn.Linear(8 * 64, 128)
        self.linear2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 8 * 64)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        return out


class WdlHead(nn.Module):
    """
    WDL head.
    """

    def __init__(self, n_hidden) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.conv = nn.Conv2d(n_hidden, 32, 1)
        self.linear1 = nn.Linear(32 * 64, 128)
        self.linear2 = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 32 * 64)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out


class SeNet(nn.Module):
    """
    ResNet model.
    """

    def __init__(
        self, n_blocks, n_hidden, n_hidden_red=32, heads=None
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.n_hidden_red = n_hidden_red

        self.ini_conv = nn.Conv2d(112, n_hidden, 3, padding=1)
        for i in range(n_blocks):
            setattr(self, f"block{i}", SeBlock(n_hidden, n_hidden_red))

        if heads is None:
            heads = ["mlh", "wdl", "policy"]
        elif "wdl" in heads and "value" in heads:
            raise ValueError("Cannot have both wdl and value heads.")
        self.heads = heads

        if "mlh" in heads:
            self.mlh = MlhHead(n_hidden)
        if "wdl" in heads:
            self.wdl = WdlHead(n_hidden)
        if "policy" in heads:
            self.policy = PolicyHead(n_hidden)
        if "value" in heads:
            self.value = ValueHead(n_hidden)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.ini_conv(x)
        out = self.relu(out)
        for i in range(self.n_blocks):
            out = getattr(self, f"block{i}")(out)

        out_dict = {}
        for head in self.heads:
            out_dict[head] = getattr(self, head)(out)

        return TensorDict(out_dict, batch_size=x.shape[0], device=x.device)

    def state_dict_mapper(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace("convert_model.", "")
            key = key.replace("/", ".")
            key = key.replace("inputconv", "ini_conv")
            if "value" not in self.heads:
                key = key.replace("value", "wdl")
            exp_ini = r"initializers\.onnx_initializer_(?P<index>\d+)"
            match = re.match(exp_ini, key)
            ignore = False
            if match:
                index = int(match.group("index"))
                if index < self.n_blocks * 6:
                    block_index = index // 6
                    if index % 6 == 0:
                        ignore = True
                    elif index % 6 == 1:
                        key = f"block{block_index}.se_layer.linear1.weight"
                        value = value.transpose(0, 1)
                    elif index % 6 == 2:
                        key = f"block{block_index}.se_layer.linear1.bias"
                    elif index % 6 == 3:
                        key = f"block{block_index}.se_layer.linear2.weight"
                        value = value.transpose(0, 1)
                    elif index % 6 == 4:
                        key = f"block{block_index}.se_layer.linear2.bias"
                    elif index % 6 == 5:
                        ignore = True
                elif index == self.n_blocks * 6 + 3:
                    key = "wdl.linear1.weight"
                    value = value.transpose(0, 1)
                elif index == self.n_blocks * 6 + 4:
                    key = "wdl.linear1.bias"
                elif index == self.n_blocks * 6 + 5:
                    key = "wdl.linear2.weight"
                    value = value.transpose(0, 1)
                elif index == self.n_blocks * 6 + 6:
                    key = "wdl.linear2.bias"
                elif index == self.n_blocks * 6 + 8:
                    key = "mlh.linear1.weight"
                    value = value.transpose(0, 1)
                elif index == self.n_blocks * 6 + 9:
                    key = "mlh.linear1.bias"
                elif index == self.n_blocks * 6 + 10:
                    key = "mlh.linear2.weight"
                    value = value.transpose(0, 1)
                elif index == self.n_blocks * 6 + 11:
                    key = "mlh.linear2.bias"
                else:
                    ignore = True
            if "wdl" not in self.heads:
                key = key.replace("wdl", "value")
            if not ignore:
                new_state_dict[key] = value
        return new_state_dict
