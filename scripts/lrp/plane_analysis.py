"""Script to compute the importance of each plane for the model.

Run with:
```
poetry run python -m scripts.lrp.plane_analysis
```
"""

import argparse
from loguru import logger

from datasets import Dataset
from torch.utils.data import DataLoader
import torch

from lczerolens.encodings import move as move_encoding
from lczerolens.concept import MulticlassConcept
from lczerolens.model import ForceValueFlow, PolicyFlow
from lczerolens import concept, LensFactory
from scripts import visualisation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dataset = Dataset.from_json(
        "./assets/TCEC_game_collection_random_boards_bestlegal.jsonl", features=MulticlassConcept.features
    )
    logger.info(f"Loaded dataset with {len(dataset)} boards.")
    if args.target == "policy":
        wrapper = PolicyFlow.from_path(f"./assets/{args.model_name}").to(DEVICE)
        init_rel_fn = concept.concept_init_rel

    elif args.target == "value":
        wrapper = ForceValueFlow.from_path(f"./assets/{args.model_name}").to(DEVICE)
        init_rel_fn = None
    else:
        raise ValueError(f"Target '{args.target}' not supported.")
    lens = LensFactory.from_name("lrp")
    if not lens.is_compatible(wrapper):
        raise ValueError(f"Lens of type 'lrp' not compatible with model '{args.model_name}'.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=concept.concept_collate_fn)

    iter_analyse = lens.analyse_batched_boards(
        dataloader,
        wrapper,
        target=None,
        return_output=True,
        init_rel_fn=init_rel_fn,
    )
    all_stats = {
        "relative_piece_relevance": [],
        "absolute_piece_relevance": [],
        "plane_relevance_proportion": [],
        "relative_piece_relevance_proportion": [],
        "absolute_piece_relevance_proportion": [],
    }
    n_plotted = 0
    for batch in iter_analyse:
        batched_relevances, boards, *infos = batch
        relevances, outputs = batched_relevances
        labels = infos[0]
        for rel, out, board, label in zip(relevances, outputs, boards, labels):
            max_config_rel = rel[:12].abs().max().item()
            if max_config_rel == 0:
                continue
            if n_plotted < args.plot_first_n:
                if board.turn:
                    heatmap = rel.sum(dim=0).view(64)
                else:
                    heatmap = rel.sum(dim=0).flip(0).view(64)
                if args.target == "policy":
                    move = move_encoding.decode_move(label, (board.turn, not board.turn), board)
                else:
                    move = None
                visualisation.render_heatmap(
                    board,
                    heatmap,
                    arrows=[(move.from_square, move.to_square)] if move is not None else None,
                    normalise="abs",
                    save_to=f"./scripts/results/{args.target}_heatmap_{n_plotted}.png",
                )
                n_plotted += 1

            plane_order = "PNBRQKpnbrqk"
            piece_relevance = {}
            for i, letter in enumerate(plane_order):
                num = (rel[i] != 0).sum().item()
                if num == 0:
                    piece_relevance[letter] = 0
                else:
                    piece_relevance[letter] = rel[i].sum().item() / num

            if args.find_interesting:
                if piece_relevance["q"] / max_config_rel > 0.9 and args.target == "value":
                    if board.turn:
                        heatmap = rel.sum(dim=0).view(64)
                    else:
                        heatmap = rel.sum(dim=0).flip(0).view(64)
                    if args.target == "policy":
                        move = move_encoding.decode_move(label, (board.turn, not board.turn), board)
                    else:
                        move = None
                    visualisation.render_heatmap(
                        board,
                        heatmap,
                        arrows=[(move.from_square, move.to_square)] if move is not None else None,
                        normalise="abs",
                        save_to=f"./scripts/results/{args.target}_heatmap_{n_plotted}.png",
                    )
                    raise SystemExit

                if any(piece_relevance[k] / max_config_rel > 0.9 for k in "pnbrqk") and args.target == "policy":
                    if board.turn:
                        heatmap = rel.sum(dim=0).view(64)
                    else:
                        heatmap = rel.sum(dim=0).flip(0).view(64)
                    if args.target == "policy":
                        move = move_encoding.decode_move(label, (board.turn, not board.turn), board)
                    else:
                        move = None
                    visualisation.render_heatmap(
                        board,
                        heatmap,
                        arrows=[(move.from_square, move.to_square)] if move is not None else None,
                        normalise="abs",
                        save_to=f"./scripts/results/{args.target}_heatmap_{n_plotted}.png",
                    )
                    raise SystemExit

            all_stats["absolute_piece_relevance"].append(piece_relevance)
            all_stats["relative_piece_relevance"].append({k: v / max_config_rel for k, v in piece_relevance.items()})

            total_relevance = rel.abs().sum().item()
            clock = board.fullmove_number * 2 - (not board.turn)
            proportion = rel.abs().sum(dim=(1, 2)).div(total_relevance).tolist()
            all_stats["plane_relevance_proportion"].append({clock: proportion})
            all_stats["relative_piece_relevance_proportion"].append(
                {clock: [v / max_config_rel for v in piece_relevance.values()]}
            )
            all_stats["absolute_piece_relevance_proportion"].append({clock: proportion[:12]})

        logger.info(f"Processed {len(all_stats['relative_piece_relevance'])} boards.")

    visualisation.render_boxplot(
        all_stats["relative_piece_relevance"],
        y_label="Relevance",
        title="Relative Relevance",
        save_to=f"./scripts/results/{args.target}_piece_relative_relevance.png",
    )
    visualisation.render_boxplot(
        all_stats["absolute_piece_relevance"],
        y_label="Relevance",
        title="Absolute Relevance",
        save_to=f"./scripts/results/{args.target}_piece_absolute_relevance.png",
    )
    visualisation.render_proportion_through_index(
        all_stats["plane_relevance_proportion"],
        plane_type="Pieces",
        y_label="Proportion of relevance",
        y_log=True,
        max_index=200,
        title="Proportion of relevance per piece",
        save_to=f"./scripts/results/{args.target}_plane_config_relevance.png",
    )
    visualisation.render_proportion_through_index(
        all_stats["plane_relevance_proportion"],
        plane_type="H0",
        y_label="Proportion of relevance",
        y_log=True,
        max_index=200,
        title="Proportion of relevance per plane",
        save_to=f"./scripts/results/{args.target}_plane_H0_relevance.png",
    )
    visualisation.render_proportion_through_index(
        all_stats["plane_relevance_proportion"],
        plane_type="Hist",
        y_label="Proportion of relevance",
        y_log=True,
        max_index=200,
        title="Proportion of relevance per plane",
        save_to=f"./scripts/results/{args.target}_plane_hist_relevance.png",
    )
    visualisation.render_proportion_through_index(
        all_stats["relative_piece_relevance_proportion"],
        plane_type="Pieces",
        y_label="Proportion of relevance",
        y_log=False,
        max_index=200,
        title="Proportion of relevance per piece",
        save_to=f"./scripts/results/{args.target}_piece_plane_relative_relevance.png",
    )
    visualisation.render_proportion_through_index(
        all_stats["absolute_piece_relevance_proportion"],
        plane_type="Pieces",
        y_label="Proportion of relevance",
        y_log=False,
        max_index=200,
        title="Proportion of relevance per piece",
        save_to=f"./scripts/results/{args.target}_piece_plane_absolute_relevance.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("plane-importance")
    parser.add_argument("--model_name", type=str, default="64x6-2018_0627_1913_08_161.onnx")
    parser.add_argument("--target", type=str, default="value")
    parser.add_argument("--find_interesting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--plot_first_n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
