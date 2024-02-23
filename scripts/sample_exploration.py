"""Script to generate datasets.

Run with:
```bash
poetry run python -m scripts.sample_exploration
```
"""

import torch
from pylatex import Document
from pylatex.package import Package

from lczerolens import ModelWrapper, move_utils
from lczerolens.xai import ConceptDataset, LrpLens
from scripts.create_figure import add_plot, create_heatmap_string

#######################################
# HYPERPARAMETERS
#######################################
model_names = {
    "396": "64x6-2018_0615_1047_23_067.onnx",
    "824": "64x6-2018_0623_0033_45_513.onnx",
    "1500": "64x6-2018_0623_0149_12_076.onnx",
    "1884": "64x6-2018_0623_0249_00_590.onnx",
    "2154": "64x6-2018_0623_0407_29_533.onnx",
    "3555 (best)": "64x6-2018_0627_1913_08_161.onnx",
}
best_legal = True
all_planes = False
target = "value"
#######################################


concept_dataset = ConceptDataset(
    "./assets/TCEC_game_collection_random_boards_bestlegal_knight_10.jsonl"
)
lens = LrpLens()

all_relevances = {}
for elo, model_name in model_names.items():
    model = ModelWrapper.from_path(f"./assets/{model_name}")
    if best_legal and target == "policy":
        label_tensor = torch.tensor(concept_dataset.labels)

        def init_rel_fn(policy):
            rel = torch.zeros_like(policy)
            rel[:, label_tensor] = policy[:, label_tensor]
            return rel

    else:
        init_rel_fn = None  # type: ignore

    def collate_fn(batch):
        indices, boards, _ = ConceptDataset.collate_fn_tuple(batch)
        return indices, boards

    relevances = lens.analyse_dataset(
        concept_dataset,
        model,
        batch_size=10,
        collate_fn=collate_fn,
        target=target,
        init_rel_fn=init_rel_fn,
    )
    all_relevances[elo] = relevances


for i, (_, board, label) in enumerate(concept_dataset):
    doc = Document(
        geometry_options={
            "lmargin": "3cm",
            "tmargin": "0.5cm",
            "bmargin": "1.5cm",
            "rmargin": "3cm",
        }
    )
    doc.packages.append(Package("xskak"))
    for elo, relevances in all_relevances.items():
        if i > 10:
            break
        move = move_utils.decode_move(
            label, (board.turn, not board.turn), board
        )
        uci_move = move.uci()
        if all_planes:
            heatmap = relevances[i].sum(dim=0)  # type: ignore
        else:
            heatmap = relevances[i][:12].sum(dim=0)  # type: ignore
        if not board.turn:
            heatmap = heatmap.view(8, 8).flip(0).view(64)
        heatmap = heatmap.view(64) / heatmap.abs().max()
        heatmap_str = create_heatmap_string(heatmap)

        doc = add_plot(
            doc,
            board.fen(),
            heatmap_str,
            current_piece_pos=uci_move[:2],
            next_move=uci_move[2:4],
            caption=f"Sample {i} - Model ELO {elo}",
        )

    doc.generate_pdf(
        "scripts/results/exploration/"
        f"{'best' if best_legal else 'full'}"
        f"_{'all' if all_planes else '12'}"
        f"_{target}_{i}",
        clean_tex=True,
    )
