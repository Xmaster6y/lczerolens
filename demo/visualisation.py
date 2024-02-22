"""
Visualisation utils.
"""

import chess
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchviz

from . import constants

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
    n_bins=20,
):
    """
    Render the policy distribution histogram.
    """
    legal_mask = torch.Tensor(
        [move in legal_moves for move in range(1858)]
    ).bool()
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    _, bins = np.histogram(policy, bins=n_bins)
    ax.hist(
        policy[~legal_mask],
        bins=bins,
        alpha=0.5,
        density=True,
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


def render_policy_statistics(
    statistics,
):
    """
    Render the policy statistics.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    move_indices = list(statistics["mean_legal_logits"].keys())
    legal_means_avg = [
        np.mean(statistics["mean_legal_logits"][move_idx])
        for move_idx in move_indices
    ]
    illegal_means_avg = [
        np.mean(statistics["mean_illegal_logits"][move_idx])
        for move_idx in move_indices
    ]
    legal_means_std = [
        np.std(statistics["mean_legal_logits"][move_idx])
        for move_idx in move_indices
    ]
    illegal_means_std = [
        np.std(statistics["mean_illegal_logits"][move_idx])
        for move_idx in move_indices
    ]
    ax.errorbar(
        move_indices,
        legal_means_avg,
        yerr=legal_means_std,
        label="Legal moves",
    )
    ax.errorbar(
        move_indices,
        illegal_means_avg,
        yerr=illegal_means_std,
        label="Illegal moves",
    )
    plt.xlabel("Move index")
    plt.ylabel("Mean policy logits")
    plt.legend()
    return fig


def render_relevance_proportion(statistics, scaled=True):
    """
    Render the relevance proportion statistics.
    """
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    fig_hist = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    move_indices = list(statistics["planes_relevance_proportion"].keys())
    for h in range(8):
        relevance_proportion_avg = [
            np.mean(
                [
                    rel[13 * h : 13 * (h + 1)].sum()
                    for rel in statistics["planes_relevance_proportion"][
                        move_idx
                    ]
                ]
            )
            for move_idx in move_indices
        ]
        relevance_proportion_std = [
            np.std(
                [
                    rel[13 * h : 13 * (h + 1)].sum()
                    for rel in statistics["planes_relevance_proportion"][
                        move_idx
                    ]
                ]
            )
            for move_idx in move_indices
        ]
        ax.errorbar(
            move_indices[h + 1 :],
            relevance_proportion_avg[h + 1 :],
            yerr=relevance_proportion_std[h + 1 :],
            label=f"History {h}",
            c=COLOR_MAP(norm(h / 9)),
        )

    relevance_proportion_avg = [
        np.mean(
            [
                rel[104:108].sum()
                for rel in statistics["planes_relevance_proportion"][move_idx]
            ]
        )
        for move_idx in move_indices
    ]
    relevance_proportion_std = [
        np.std(
            [
                rel[104:108].sum()
                for rel in statistics["planes_relevance_proportion"][move_idx]
            ]
        )
        for move_idx in move_indices
    ]
    ax.errorbar(
        move_indices,
        relevance_proportion_avg,
        yerr=relevance_proportion_std,
        label="Castling rights",
        c=COLOR_MAP(norm(8 / 9)),
    )
    relevance_proportion_avg = [
        np.mean(
            [
                rel[108:].sum()
                for rel in statistics["planes_relevance_proportion"][move_idx]
            ]
        )
        for move_idx in move_indices
    ]
    relevance_proportion_std = [
        np.std(
            [
                rel[108:].sum()
                for rel in statistics["planes_relevance_proportion"][move_idx]
            ]
        )
        for move_idx in move_indices
    ]
    ax.errorbar(
        move_indices,
        relevance_proportion_avg,
        yerr=relevance_proportion_std,
        label="Remaining planes",
        c=COLOR_MAP(norm(9 / 9)),
    )
    plt.xlabel("Move index")
    plt.ylabel("Absolute relevance proportion")
    plt.yscale("log")
    plt.legend()

    if scaled:
        stat_key = "planes_relevance_proportion_scaled"
    else:
        stat_key = "planes_relevance_proportion"
    fig_planes = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    move_indices = list(statistics[stat_key].keys())
    for p in range(13):
        relevance_proportion_avg = [
            np.mean([rel[p].item() for rel in statistics[stat_key][move_idx]])
            for move_idx in move_indices
        ]
        relevance_proportion_std = [
            np.std([rel[p].item() for rel in statistics[stat_key][move_idx]])
            for move_idx in move_indices
        ]
        ax.errorbar(
            move_indices,
            relevance_proportion_avg,
            yerr=relevance_proportion_std,
            label=constants.PLANE_NAMES[p],
            c=COLOR_MAP(norm(p / 12)),
        )

    plt.xlabel("Move index")
    plt.ylabel("Absolute relevance proportion")
    plt.yscale("log")
    plt.legend()

    fig_pieces = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for p in range(1, 13):
        stat_key = f"configuration_relevance_proportion_threatened_piece{p}"
        n_attackers = list(statistics[stat_key].keys())
        relevance_proportion_avg = [
            np.mean(
                statistics[
                    f"configuration_relevance_proportion_threatened_piece{p}"
                ][n]
            )
            for n in n_attackers
        ]
        relevance_proportion_std = [
            np.std(statistics[stat_key][n]) for n in n_attackers
        ]
        ax.errorbar(
            n_attackers,
            relevance_proportion_avg,
            yerr=relevance_proportion_std,
            label="PNBRQKpnbrqk"[p - 1],
            c=COLOR_MAP(norm(p / 12)),
        )

    plt.xlabel("Number of attackers")
    plt.ylabel("Absolute configuration relevance proportion")
    plt.yscale("log")
    plt.legend()

    return fig_hist, fig_planes, fig_pieces


def render_probing_statistics(
    statistics,
):
    """
    Render the probing statistics.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    n_blocks = len(statistics["metrics"])
    for metric in statistics["metrics"]["block0"]:
        avg = []
        std = []
        for block_idx in range(n_blocks):
            metrics = statistics["metrics"]
            block_data = metrics[f"block{block_idx}"]
            avg.append(np.mean(block_data[metric]))
            std.append(np.std(block_data[metric]))
        ax.errorbar(
            range(n_blocks),
            avg,
            yerr=std,
            label=metric,
        )
    plt.xlabel("Block index")
    plt.ylabel("Metric")
    plt.yscale("log")
    plt.legend()
    return fig
