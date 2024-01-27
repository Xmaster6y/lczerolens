"""
Visualisation utils.
"""

import chess
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchviz

COLOR_MAP = matplotlib.colormaps["RdYlBu_r"].resampled(1000)
ALPHA = 1.0


def render_heatmap(
    board,
    heatmap,
    square=None,
    vmin=None,
    vmax=None,
    arrows=None,
    normalise="none",
):
    """
    Render a heatmap on the board.
    """
    if normalise == "abs":
        a_max = heatmap.abs().max()
        if a_max != 0:
            heatmap = heatmap / a_max
        vmin = -1
        vmax = 1
    if vmin is None:
        vmin = heatmap.min()
    if vmax is None:
        vmax = heatmap.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    color_dict = {}
    for square_index in range(64):
        color = COLOR_MAP(norm(heatmap[square_index]))
        color = (*color[:3], ALPHA)
        color_dict[square_index] = matplotlib.colors.to_hex(
            color, keep_alpha=True
        )
    fig = plt.figure(figsize=(6, 0.6))
    ax = plt.gca()
    ax.axis("off")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=COLOR_MAP),
        ax=ax,
        orientation="horizontal",
        fraction=1.0,
    )
    if square is not None:
        try:
            check = chess.parse_square(square)
        except ValueError:
            check = None
    else:
        check = None
    if arrows is None:
        arrows = []
    plt.close()
    return (
        chess.svg.board(
            board,
            check=check,
            fill=color_dict,
            size=350,
            arrows=arrows,
        ),
        fig,
    )


def render_architecture(model, name: str = "model", directory: str = ""):
    """
    Render the architecture of the model.
    """
    out = model(torch.zeros(1, 112, 8, 8))
    if len(out) == 2:
        policy, outcome_probs = out
        value = torch.zeros(outcome_probs.shape[0], 1)
    else:
        policy, outcome_probs, value = out
    torchviz.make_dot(
        policy, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_policy", format="svg")
    torchviz.make_dot(
        outcome_probs, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_outcome_probs", format="svg")
    torchviz.make_dot(
        value, params=dict(list(model.named_parameters()))
    ).render(f"{directory}/{name}_value", format="svg")


def render_policy_distribution(
    policy,
    legal_moves,
    n_bins=10,
):
    """
    Render the policy distribution histogram.
    """
    legal_mask = torch.Tensor(
        [move in legal_moves for move in range(1858)]
    ).bool()
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    _, bins, _ = ax.hist(
        policy[~legal_mask],
        bins=n_bins,
        density=True,
        alpha=0.5,
        label="Illegal moves",
    )
    ax.hist(
        policy[legal_mask],
        bins=bins,
        alpha=0.5,
        density=True,
        label="Legal moves",
    )
    plt.xlabel("Policy")
    plt.ylabel("Density")
    plt.legend()
    plt.yscale("log")
    return fig
