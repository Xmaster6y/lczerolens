"""
Custom ViT network.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from tensordict import TensorDict
from torch import nn

from . import constants
from .network import MultiplyLayer, ProdLayer, SofplusTanhMul, SumLayer


class SmolgenLayer(nn.Module):
    """
    Smolgen layer.
    """

    def __init__(
        self, n_hidden, n_heads, n_hidden_red, smolgen_decompress
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_hidden_red = n_hidden_red

        self.compress = nn.Linear(n_hidden, 32, bias=False)
        self.linear1 = nn.Linear(32 * 64, n_hidden_red)
        self.ln1 = nn.LayerNorm(n_hidden_red)
        self.linear2 = nn.Linear(n_hidden_red, n_hidden_red * n_heads)
        self.ln2 = nn.LayerNorm(n_hidden_red * n_heads)
        self.decompress = smolgen_decompress

        self.prod_layer = ProdLayer(dim=-1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.compress(x)
        out = out.view(-1, 32 * 64)
        out = self.linear1(out)
        non_lin = self.sigmoid(out).detach()
        out = self.prod_layer(torch.stack([out, non_lin], dim=-1))
        out = self.ln1(out)
        out = self.linear2(out)
        non_lin = self.sigmoid(out).detach()
        out = self.prod_layer(torch.stack([out, non_lin], dim=-1))
        out = self.ln2(out)
        out = out.view(-1, self.n_heads, self.n_hidden_red)
        out = torch.matmul(out, self.decompress)
        return out.view(-1, self.n_heads, 64, 64)


class MlpLayer(nn.Module):
    """
    MLP layer.
    """

    def __init__(self, n_hidden, scale) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.scale = scale

        self.linear1 = nn.Linear(n_hidden, 1024)
        self.linear2 = nn.Linear(1024, n_hidden)

        self.sum_layer = SumLayer(dim=-1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return self.sum_layer(torch.stack([out, x * self.scale], dim=-1))


class EncoderBlock(nn.Module):
    """
    Custom encoder block.
    """

    def __init__(
        self,
        n_hidden,
        n_heads,
        mlp_scale,
        attention_scale,
        n_hidden_red,
        smolgen_decompress,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.n_hidden_red = n_hidden_red
        self.attention_scale = attention_scale
        self.mlp_scale = mlp_scale

        self.q_proj = nn.Linear(n_hidden, n_heads * 64)
        self.k_proj = nn.Linear(n_hidden, n_heads * 64)
        self.v_proj = nn.Linear(n_hidden, n_heads * 64)
        self.out_proj = nn.Linear(n_heads * 64, n_hidden)
        self.smolgen = SmolgenLayer(
            n_hidden, n_heads, n_hidden_red, smolgen_decompress
        )
        self.ln1 = nn.LayerNorm(n_hidden)
        self.mlp = MlpLayer(n_hidden, mlp_scale)
        self.ln2 = nn.LayerNorm(n_hidden)

        self.sum_layer = SumLayer(dim=-1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q_proj(x).view(-1, 64, self.n_heads, 64).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(-1, 64, self.n_heads, 64).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(-1, 64, self.n_heads, 64).permute(0, 2, 1, 3)
        qk = torch.matmul(q, k) * self.attention_scale
        smolgen_out = self.smolgen(x)
        qk = self.sum_layer(torch.stack([qk, smolgen_out], dim=-1))
        qk = self.softmax(qk)
        out = (
            torch.matmul(qk, v)
            .permute(0, 2, 1, 3)
            .reshape(-1, 64, self.n_heads * 64)
        )
        out = self.out_proj(out)
        out = self.sum_layer(torch.stack([out, x * self.mlp_scale], dim=-1))
        out = self.ln1(out)
        out = self.mlp(out)
        out = self.ln2(out)
        return out


class PolicyHead(nn.Module):
    """
    Policy head.
    """

    def __init__(self, n_hidden, scale, act_mode) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.scale = scale

        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_hidden)
        self.linear4 = nn.Linear(n_hidden, 4, bias=False)

        self.prod_layer = ProdLayer(dim=-1)
        self.sum_layer = SumLayer(dim=-1)

        self.act_mode = act_mode
        if act_mode == "relu":
            self.act = nn.ReLU()
        elif act_mode == "softplus-tanh-mul":
            self.act = SofplusTanhMul()
        else:
            raise ValueError(f"Unknown activation mode {act_mode}.")

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out1 = self.linear2(out).view(-1, 64, self.n_hidden)
        out2 = self.linear3(out).view(-1, 64, self.n_hidden)
        out1 = torch.matmul(out1, out2.permute(0, 2, 1)) * self.scale
        out2 = out2[:, 56:64, :]
        out2 = self.linear4(out2).permute(0, 2, 1)
        chunk1, chunk2 = out1.split((3, 1), dim=-1)
        out2 = self.sum_layer(torch.concat([chunk1, chunk2], dim=-1))
        out2 = out2.permute(0, 2, 1).view(-1, 1, 24)
        out3 = out1[:, 48:56, 56:64].view(-1, 64, 1)
        out3 = out3.view(-1, 8, 24)
        out2 = self.sum_layer(torch.stack([out2, out3], dim=-1))
        out2 = out2.view(-1, 3, 64)
        out = torch.concat([out1, out2], dim=1)
        out = out.view(-1, 4288)
        out = out.gather(
            1,
            torch.tensor(constants.GATHER_INDICES)
            .unsqueeze(0)
            .repeat(out.shape[0], 1)
            .to(out.device),
        )
        return out


class MlhHead(nn.Module):
    """
    MLH head.
    """

    def __init__(
        self,
        n_hidden,
        n_hidden_red,
        act_mode,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_hidden_red = n_hidden_red

        self.embed = nn.Linear(n_hidden, n_hidden_red)
        self.linear1 = nn.Linear(n_hidden_red * 64, 128)
        self.linear2 = nn.Linear(128, 1)

        self.act_mode = act_mode
        if act_mode == "relu":
            self.act = nn.ReLU()
        elif act_mode == "softplus-tanh-mul":
            self.act = SofplusTanhMul()
        else:
            raise ValueError(f"Unknown activation mode {act_mode}.")

    def forward(self, x):
        out = self.embed(x)
        out = self.act(out)
        out = out.view(-1, 64 * self.n_hidden_red)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return self.act(out)


class WdlHead(nn.Module):
    """
    WDL head.
    """

    def __init__(
        self,
        n_hidden,
        n_hidden_red,
        act_mode,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_hidden_red = n_hidden_red

        self.embed = nn.Linear(n_hidden, n_hidden_red)
        self.linear1 = nn.Linear(n_hidden_red * 64, 128)
        self.linear2 = nn.Linear(128, 3)

        self.act_mode = act_mode
        if act_mode == "relu":
            self.act = nn.ReLU()
        elif act_mode == "softplus-tanh-mul":
            self.act = SofplusTanhMul()
        else:
            raise ValueError(f"Unknown activation mode {act_mode}.")
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.embed(x)
        out = self.act(out)
        out = out.view(-1, 64 * self.n_hidden_red)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return self.softmax(out)


@dataclass
class VitConfig:
    n_blocks: int
    n_hidden: int
    n_heads: int
    n_hidden_red_block: int
    smolgen_decompress: torch.Tensor
    positional_encodings: torch.Tensor
    mlp_scale: float = 1.0
    attention_scale: float = 1.0
    policy_scale: float = 1.0
    n_hidden_red_mlh: int = 8
    n_hidden_red_wdl: int = 64
    act_mode: str = "relu"
    heads: Optional[List[str]] = None


class VitNet(nn.Module):
    """
    Custom transformer model.
    """

    def __init__(
        self,
        config: VitConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.ini_linear = nn.Linear(176, config.n_hidden)
        self.ini_multiply = MultiplyLayer((64, config.n_hidden))
        for i in range(config.n_blocks):
            setattr(
                self,
                f"block{i}",
                EncoderBlock(
                    config.n_hidden,
                    config.n_heads,
                    config.mlp_scale,
                    config.attention_scale,
                    config.n_hidden_red_block,
                    config.smolgen_decompress,
                ),
            )

        if config.heads is None:
            config.heads = ["mlh", "wdl", "policy"]
        elif "value" in config.heads:
            raise NotImplementedError("Value head not implemented.")
        self.heads = config.heads

        if "mlh" in config.heads:
            self.mlh = MlhHead(
                config.n_hidden, config.n_hidden_red_mlh, config.act_mode
            )
        if "wdl" in config.heads:
            self.wdl = WdlHead(
                config.n_hidden, config.n_hidden_red_wdl, config.act_mode
            )
        if "policy" in config.heads:
            self.policy = PolicyHead(
                config.n_hidden, config.policy_scale, config.act_mode
            )

        self.act_mode = config.act_mode
        if config.act_mode == "relu":
            self.act = nn.ReLU()
        elif config.act_mode == "softplus-tanh-mul":
            self.act = SofplusTanhMul()
        else:
            raise ValueError(f"Unknown activation mode {config.act_mode}.")

    def forward(self, x):
        out = x.permute(0, 2, 3, 1).view(-1, 64, 112)
        positional_encodings = self.config.positional_encodings.view(
            1, 64, 64
        ).repeat((out.shape[0], 1, 1))
        out = torch.cat([out, positional_encodings], dim=-1)
        out = self.ini_linear(out)
        out = self.act(out)
        out = out.view(-1, 64, self.config.n_hidden)
        out = self.ini_multiply(out)
        for i in range(self.config.n_blocks):
            out = getattr(self, f"block{i}")(out)

        out_dict = {}
        for head in self.heads:
            if head == "policy":
                warnings.warn("Policy head is not yet implemented.")
                out_dict[head] = torch.zeros(
                    (x.shape[0], 1858), device=x.device
                )
                continue
            out_dict[head] = getattr(self, head)(out)
        return TensorDict(out_dict, batch_size=x.shape[0], device=x.device)
