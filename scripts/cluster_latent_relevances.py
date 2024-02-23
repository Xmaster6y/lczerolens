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
from pylatex import Document
from pylatex.package import Package
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from lczerolens import move_utils
from lczerolens.game import PolicyFlow
from lczerolens.xai import ConceptDataset, LrpLens
from scripts.create_figure import add_plot, create_heatmap_string

#######################################
# HYPERPARAMETERS
#######################################
n_clusters = 10
batch_size = 500
save_files = False
model_name = "64x6-2018_0627_1913_08_161.onnx"
dataset_name = "TCEC_game_collection_random_boards_bestlegal_knight.jsonl"
only_config_rel = True
best_legal = True
run_name = (
    f"bestres_tcec_bestlegal_knight_{'expbest' if best_legal else 'full'}"
)
conv_sum_dims = ()
cosine_sim = False
kmeans_on_tsne = True
viz_name = (
    f"nosum_{'cosine' if cosine_sim else 'norm'}"
    f"_{'after' if kmeans_on_tsne else 'before'}-tsne"
)
#######################################


def legal_init_rel(board_list, board_tensor):
    legal_move_mask = torch.zeros((len(board_list), 1858))
    for idx, board in enumerate(board_list):
        legal_moves = [
            move_utils.encode_move(move, (board.turn, not board.turn))
            for move in board.legal_moves
        ]
        legal_move_mask[idx, legal_moves] = 1
    return legal_move_mask * board_tensor


model = PolicyFlow.from_path(f"./assets/{model_name}")
concept_dataset = ConceptDataset(f"./assets/{dataset_name}")
print(f"[INFO] Board dataset len: {len(concept_dataset)}")

composite = LrpLens.make_default_composite()
layer_names = [f"model.block{b}/conv2/relu" for b in [0, 3, 5]]
print(layer_names)

dataloader = torch.utils.data.DataLoader(
    concept_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=ConceptDataset.collate_fn_tensor,
)

if save_files:
    print("############ Collecting Relevances")
    all_relevances = {}
    for batch in tqdm(dataloader):
        _, board_tensor, labels = batch
        label_tensor = torch.tensor(labels)

        def init_rel_fn(board_tensor):
            rel = torch.zeros_like(board_tensor)
            rel[:, label_tensor] = board_tensor[:, label_tensor]
            return rel

        board_tensor.requires_grad = True
        with LrpLens.context(model) as modifed_model:
            attribution = CondAttribution(modifed_model)
            attr = attribution(
                board_tensor,
                [{"y": None}],
                composite,
                record_layer=layer_names,
                init_rel=init_rel_fn if best_legal else None,
            )

            for layer_name in layer_names:
                latent_rel = attr.relevances[layer_name]
                if layer_name not in all_relevances:
                    all_relevances[layer_name] = latent_rel.detach().cpu()
                else:
                    all_relevances[layer_name] = torch.cat(
                        [
                            all_relevances[layer_name],
                            latent_rel.detach().cpu(),
                        ],
                        dim=0,
                    )

    os.makedirs(f"scripts/clusters/{run_name}", exist_ok=True)
    save_file(
        all_relevances,
        f"scripts/clusters/{run_name}/relevances.safetensors",
    )

else:
    all_relevances = {}
    with safe_open(
        f"scripts/clusters/{run_name}/relevances.safetensors",
        framework="pt",
        device="cpu",
    ) as f:
        for key in f.keys():
            all_relevances[key] = f.get_tensor(key)

#######################################
# Cluster the latent relevances
#######################################

print("############ Clustering ...")
os.makedirs(f"scripts/results/{run_name}/{viz_name}", exist_ok=True)

for layer_name, relevances in all_relevances.items():
    if conv_sum_dims:
        relevances = relevances.sum(dim=conv_sum_dims).view(
            relevances.shape[0], -1
        )
    else:
        relevances = relevances.view(relevances.shape[0], -1)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2)
    latent_rel_tsne = tsne.fit_transform(relevances)

    if kmeans_on_tsne:
        relevances = latent_rel_tsne
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
    kmeans.fit(relevances)

    # Plot the clustered data
    plt.scatter(latent_rel_tsne[:, 0], latent_rel_tsne[:, 1], c=kmeans.labels_)
    plt.title("Clustered Latent Relevances")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(
        f"scripts/results/{run_name}/{viz_name}/"
        f"{layer_name.replace('/','.')}_t-sne.png"
    )
    plt.close()

    #######################################
    # Plot chessboards for each cluster
    #######################################

    print("############ Plotting chessboards for each cluster")
    with LrpLens.context(model) as modifed_model:
        attribution = CondAttribution(modifed_model)
        for idx_cluster in tqdm(range(n_clusters)):
            cluster_center = kmeans.cluster_centers_[idx_cluster]
            if cosine_sim:
                dot_prod = relevances @ cluster_center.T
                similarities = dot_prod / (
                    np.linalg.norm(relevances, axis=1)
                    * np.linalg.norm(cluster_center)
                )
                nearest_neighbors = np.argsort(similarities)[-8:]
            else:
                distances = np.linalg.norm(relevances - cluster_center, axis=1)
                nearest_neighbors = np.argsort(distances)[:8]

            doc = Document()  # create a new document
            doc.packages.append(Package("xskak"))

            # compute heatmap for each nearest neighbor
            for idx_sample in nearest_neighbors:
                _, board, label = concept_dataset[idx_sample]
                _, board_tensor, _ = ConceptDataset.collate_fn_tensor(
                    [concept_dataset[idx_sample]]
                )
                label_tensor = torch.tensor([label])

                def init_rel_fn(board_tensor):
                    rel = torch.zeros_like(board_tensor)
                    rel[:, label_tensor] = board_tensor[:, label_tensor]
                    return rel

                board_tensor.requires_grad = True
                attr = attribution(
                    board_tensor,
                    [{"y": None}],
                    composite,
                    init_rel=init_rel_fn if best_legal else None,
                )
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
                f"scripts/results/{run_name}"
                f"/{viz_name}/{layer_name.replace('/','.')}"
                f"_cluster_{idx_cluster}",
                clean_tex=True,
            )
