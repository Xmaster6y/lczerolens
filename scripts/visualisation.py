"""
Visualisation utils.
"""

import chess
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lczerolens.encodings import board as board_encoding

COLOR_MAP = matplotlib.colormaps["RdYlBu_r"].resampled(1000)
ALPHA = 1.0
NORM = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)


def render_heatmap(
    board,
    heatmap,
    square=None,
    vmin=None,
    vmax=None,
    arrows=None,
    normalise="none",
    save_to=None,
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
        color_dict[square_index] = matplotlib.colors.to_hex(color, keep_alpha=True)
    fig = plt.figure(figsize=(1, 6))
    ax = plt.gca()
    ax.axis("off")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=COLOR_MAP),
        ax=ax,
        orientation="vertical",
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

    svg_board = chess.svg.board(
        board,
        check=check,
        fill=color_dict,
        size=350,
        arrows=arrows,
    )
    if save_to is not None:
        plt.savefig(save_to)
        with open(save_to.replace(".png", ".svg"), "w") as f:
            f.write(svg_board)
        plt.close()
    else:
        plt.close()
        return svg_board, fig


def render_boxplot(
    data,
    filter_null=True,
    y_label=None,
    title=None,
    save_to=None,
):
    labels = data[0].keys()
    boxed_data = {label: [] for label in labels}
    for d in data:
        for label in labels:
            v = d.get(label)
            if v == 0.0 and filter_null:
                continue
            boxed_data[label].append(v)
    plt.boxplot(boxed_data.values(), notch=True, vert=True, patch_artist=True, labels=labels)
    plt.ylabel(y_label)
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def render_proportion_through_index(
    data,
    plane_type="H0",
    max_index=None,
    y_log=False,
    y_label=None,
    title=None,
    save_to=None,
):
    if plane_type == "H0":
        indexed_data = {
            "H0": {},
            "Hist": {},
            "Meta": {},
        }
        for d in data:
            index, proportion = next(iter(d.items()))
            if max_index is not None and index > max_index:
                continue
            if index not in indexed_data["H0"]:
                indexed_data["H0"][index] = [sum(proportion[:13])]
                indexed_data["Hist"][index] = [sum(proportion[13:104])]
                indexed_data["Meta"][index] = [sum(proportion[104:])]
            else:
                indexed_data["H0"][index].append(sum(proportion[:13]))
                indexed_data["Hist"][index].append(sum(proportion[13:104]))
                indexed_data["Meta"][index].append(sum(proportion[104:]))

    elif plane_type == "Hist":
        indexed_data = {
            "H0": {},
            "H1": {},
            "H2": {},
            "H3": {},
            "H4": {},
            "H5": {},
            "H6": {},
            "H7": {},
            "Castling": {},
            "Remaining": {},
        }
        for d in data:
            index, proportion = next(iter(d.items()))
            if max_index is not None and index > max_index:
                continue
            if index not in indexed_data["H0"]:
                for i in range(8):
                    indexed_data[f"H{i}"][index] = [sum(proportion[13 * i : 13 * (i + 1)])]
                indexed_data["Castling"][index] = [sum(proportion[104:108])]
                indexed_data["Remaining"][index] = [sum(proportion[108:])]
            else:
                for i in range(8):
                    indexed_data[f"H{i}"][index].append(sum(proportion[13 * i : 13 * (i + 1)]))
                indexed_data["Castling"][index].append(sum(proportion[104:108]))
                indexed_data["Remaining"][index].append(sum(proportion[108:]))

    elif plane_type == "Pieces":
        relative_plane_order = board_encoding.get_plane_order((chess.WHITE, chess.BLACK))
        indexed_data = {letter: {} for letter in relative_plane_order}
        for d in data:
            index, proportion = next(iter(d.items()))
            if max_index is not None and index > max_index:
                continue
            if index not in indexed_data[relative_plane_order[0]]:
                for i, letter in enumerate(relative_plane_order):
                    indexed_data[letter][index] = [proportion[i]]
            else:
                for i, letter in enumerate(relative_plane_order):
                    indexed_data[letter][index].append(proportion[i])
    else:
        raise ValueError(f"Invalid plane type: {plane_type}")

    n_curves = len(indexed_data)
    for i, (label, curve_data) in enumerate(indexed_data.items()):
        indices = sorted(list(curve_data.keys()))
        mean_curve = [np.mean(curve_data[idx]) for idx in indices]
        std_curve = [np.std(curve_data[idx]) for idx in indices]
        c = COLOR_MAP(i / (n_curves - 1))
        plt.plot(indices, mean_curve, label=label, c=c)
        lower_bound = np.array(mean_curve) - np.array(std_curve)
        upper_bound = np.array(mean_curve) + np.array(std_curve)
        plt.fill_between(indices, lower_bound, upper_bound, alpha=0.2, color=c)
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.ylabel(y_label)
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
