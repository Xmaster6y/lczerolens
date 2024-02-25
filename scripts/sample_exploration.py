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
target = "value"
n_samples = 3
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
    if i >= n_samples:
        break
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
        move = move_utils.decode_move(
            label, (board.turn, not board.turn), board
        )
        uci_move = move.uci()
        input_relevances = relevances[i]  # type: ignore
        if not board.turn:
            input_relevances = input_relevances.flip(1)
        input_relevances = input_relevances.view(112, 64)
        heatmap_str_list = [
            create_heatmap_string(input_relevances.sum(dim=0), abs_max=True),
            create_heatmap_string(
                input_relevances[:12].sum(dim=0), abs_max=True
            ),
            create_heatmap_string(
                input_relevances[104:].sum(dim=0), abs_max=True
            ),
        ]
        heatmap_caption_list = [
            "Total relevance",
            "Current config relevance",
            "Meta relevance",
        ]

        h0 = input_relevances[:13].abs().sum()
        hist = input_relevances[13:104].abs().sum()
        meta = input_relevances[104:].abs().sum()
        total = (h0 + hist + meta) / 100

        doc = add_plot(
            doc,
            board.fen(),
            heatmap_str_list,
            current_piece_pos=uci_move[:2],
            next_move=uci_move[2:4],
            caption=f"Sample {i} - Model ELO {elo} "
            f"- {h0/total:.0f}%|{hist/total:.0f}%|{meta/total:.0f}%",
            heatmap_caption_list=heatmap_caption_list,
        )

    doc.generate_pdf(
        "scripts/results/exploration/"
        f"{'best' if best_legal else 'full'}"
        f"_{target}_{i}",
        clean_tex=True,
    )
