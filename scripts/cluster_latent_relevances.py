"""Script to cluster the latent relevances of the model for a given dataset.

Run with:
```bash
poetry run python -m scripts.cluster_latent_relevances
```
"""

import os

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from pylatex import Document
from pylatex.package import Package
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from lczerolens import GameDataset, move_utils
from lczerolens.adapt import PolicyFlow
from lczerolens.xai import LrpLens, UniqueConceptDataset
from lczerolens.xai.concepts import BestLegalMoveConcept
from scripts.create_figure import add_plot, create_heatmap_string

#######################################
# HYPERPARAMETERS
#######################################
n_clusters = 10
layer_index = -1
batch_size = 500
save_files = False
conv_sum_dims = (2, 3)
model_name = "tinygyal-8.onnx"
dataset_name = "test_stockfish_10.jsonl"
only_config_rel = True
#######################################


class MaxLogitFlow(PolicyFlow):
    def forward(self, x):
        policy = super().forward(x)
        return policy.max(dim=1, keepdim=True).values


model = MaxLogitFlow.from_path(f"./assets/{model_name}")
dataset = GameDataset(f"./assets/{dataset_name}")
concept = BestLegalMoveConcept(model)
unique_dataset = UniqueConceptDataset.from_game_dataset(dataset, concept)
print(f"[INFO] Board dataset len: {len(unique_dataset)}")

composite = LrpLens.make_default_composite()
attribution = CondAttribution(model)
cc = ChannelConcept()

layer_names = get_layer_names(model, [torch.nn.ReLU])
layer_names = [
    layer_name for layer_name in layer_names if "block" in layer_name
]
print(layer_names)

dataloader = torch.utils.data.DataLoader(
    unique_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=UniqueConceptDataset.collate_fn_tensor,
)

if save_files:
    print("############ Collecting Relevances")
    all_relevances = {}
    for batch in tqdm(dataloader):
        _, board_tensor, _ = batch
        board_tensor.requires_grad = True
        attr = attribution(
            board_tensor, [{"y": 0}], composite, record_layer=layer_names
        )

        for layer_name in layer_names:
            latent_rel = attr.relevances[layer_name]
            latent_rel = cc.attribute(latent_rel, abs_norm=True)
            if len(latent_rel.shape) == 4:
                latent_rel = latent_rel.sum(conv_sum_dims)
            if layer_name not in all_relevances:
                all_relevances[layer_name] = latent_rel.detach().cpu()
            else:
                all_relevances[layer_name] = torch.cat(
                    [all_relevances[layer_name], latent_rel.detach().cpu()],
                    dim=0,
                )

    os.makedirs(f"scripts/clusters/{model_name}-{dataset_name}", exist_ok=True)
    save_file(
        all_relevances,
        f"scripts/clusters/{model_name}-{dataset_name}/relevances.safetensors",
    )

else:
    all_relevances = {}
    with safe_open(
        f"scripts/clusters/{model_name}-{dataset_name}/relevances.safetensors",
        framework="pt",
        device="cpu",
    ) as f:
        for key in f.keys():
            all_relevances[key] = f.get_tensor(key)

#######################################
# Cluster the latent relevances
#######################################

print("############ Clustering ...")
os.makedirs(f"scripts/results/{model_name}-{dataset_name}", exist_ok=True)

for layer_name, relevances in all_relevances.items():
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
    kmeans.fit(relevances)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2)
    latent_rel_tsne = tsne.fit_transform(relevances)

    # Plot the clustered data
    plt.scatter(latent_rel_tsne[:, 0], latent_rel_tsne[:, 1], c=kmeans.labels_)
    plt.title("Clustered Latent Relevances")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(
        f"scripts/results/{model_name}-{dataset_name}/{layer_name}_t-sne.png"
    )
    plt.close()

    #######################################
    # Plot chessboards for each cluster
    #######################################

    print("############ Plotting chessboards for each cluster")

    for idx_cluster in tqdm(range(n_clusters)):
        cluster_center = kmeans.cluster_centers_[idx_cluster]
        distances = np.linalg.norm(relevances - cluster_center, axis=1)
        nearest_neighbors = np.argsort(distances)[:10]

        doc = Document()  # create a new document
        doc.packages.append(Package("xskak"))

        # compute heatmap for each nearest neighbor
        for idx_sample in nearest_neighbors:
            _, board, label = unique_dataset[idx_sample]
            _, board_tensor, _ = UniqueConceptDataset.collate_fn_tensor(
                [unique_dataset[idx_sample]]
            )
            board_tensor.requires_grad = True
            attr = attribution(board_tensor, [{"y": 0}], composite)
            if only_config_rel:
                heatmap = board_tensor.grad[0, :12].sum(dim=0).view(64)
            else:
                heatmap = board_tensor.grad[0].sum(dim=0).view(64)
            if board.turn == chess.BLACK:
                heatmap = heatmap.view(8, 8).flip(0).view(64)
            move = move_utils.decode_move(
                label, (board.turn, not board.turn), board
            )
            uci_move = move.uci()
            heatmap = heatmap / heatmap.abs().max()
            heatmap_str = create_heatmap_string(heatmap)

            doc = add_plot(
                doc,
                board.fen(),
                heatmap_str,
                current_piece_pos=uci_move[:2],
                next_move=uci_move[2:4],
            )

        # Generate pdf
        doc.generate_pdf(
            f"scripts/results/{model_name}-{dataset_name}"
            f"/{layer_name}_cluster_{idx_cluster}",
            clean_tex=True,
        )
