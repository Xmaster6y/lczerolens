"""Script to find concepts in a model using LRP and a dataset of boards.

Run with:
```bash
poetry run python -m scripts.find_concepts
```
"""

import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from lczerolens import GameDataset
from lczerolens.adapt import PolicyFlow
from lczerolens.xai import LrpLens, UniqueConceptDataset
from lczerolens.xai.concepts import (
    HasMaterialAdvantageConcept,
    HasMateThreatConcept,
    HasThreatConcept,
)
from lczerolens.xai.helpers import ModifiedFeatureVisualization

#######################################
# HYPERPARAMETERS
#######################################
topk = 5
ref_mode = "activation"
batch_size = 500
save_files = False
model_name = "tinygyal-8.onnx"
dataset_name = "test_stockfish_10.jsonl"
#######################################


model = PolicyFlow.from_path(f"./assets/{model_name}")
dataset = GameDataset(f"./assets/{dataset_name}")
check_concept = HasThreatConcept("K", relative=True)
unique_dataset = UniqueConceptDataset.from_game_dataset(dataset, check_concept)
print(f"[INFO] Board dataset len: {len(unique_dataset)}")


def get_n_concepts(l_name, model):
    n_concepts = None
    for name, layer in model.named_modules():
        if l_name == name:
            n_concepts = layer.out_channels

    if n_concepts is None:
        raise ValueError(f"Layer {l_name} not found in model")
    return n_concepts


composite = LrpLens.make_default_composite()
attribution = CondAttribution(model)
cc = ChannelConcept()
layer_names = get_layer_names(model, [torch.nn.Conv2d])
layer_map = {layer: cc for layer in layer_names}


fv_path = f"scripts/im_viz/{model_name}-{dataset_name}"
fv = ModifiedFeatureVisualization(
    attribution, unique_dataset, layer_map, preprocess_fn=None, path=fv_path
)


def collate_fn_tensor(batch):
    _, board_tensor, targets = UniqueConceptDataset.collate_fn_tensor(batch)
    board_tensor.requires_grad = True
    return board_tensor, targets


def collate_fn_tuple(batch):
    _, boards, targets = UniqueConceptDataset.collate_fn_tuple(batch)
    return boards, targets


if save_files:
    saved_files = fv.run(
        composite, batch_size, 100, custom_collate_fn=collate_fn_tensor
    )
    print("[INFO] Files saved!")

concepts = {
    "in_check": HasThreatConcept("K", relative=True),
    "threat_opp_queen": HasThreatConcept("q", relative=True),
    "has_mate_threat": HasMateThreatConcept(),
    "material_advantage": HasMaterialAdvantageConcept(relative=True),
}
for case, concept in concepts.items():
    unique_dataset.concept = concept

    concept_fen_strings = set(
        [b.fen() for _, b, label in unique_dataset if label == 1]
    )
    print(f"[INFO] Concept '{case}' positives: {len(concept_fen_strings)}")

    for l_name in layer_names:
        n_concepts = get_n_concepts(l_name, model)
        intersections = []

        for i in range(n_concepts):
            ref_c = fv.get_max_reference(
                i,
                l_name,
                ref_mode,
                plot_fn=None,
                r_range=(0, -1),
                batch_size=batch_size,
                custom_collate_fn=collate_fn_tuple,
            )

            boards = ref_c[i][0]
            fen_strings = set([b.fen() for b in boards])

            # compute intersection
            intersec = concept_fen_strings.intersection(fen_strings)
            percentage = len(intersec) / len(fen_strings)
            intersections.append(percentage)

        # get topk intersection and print
        intersections = torch.tensor(intersections)
        topk_values, topk_concepts = torch.topk(intersections, topk, dim=0)
        print("### Layer:", l_name, "Case:", case, "###")
        for c, v in zip(topk_concepts, topk_values):
            print(f"Concept {c} with intersection {v*100:.0f} %")

print("[INFO] Analysis done!")
