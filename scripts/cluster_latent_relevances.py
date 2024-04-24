"""Script to cluster the latent relevances of the model for a given dataset.

Run with:
```bash
poetry run python -m scripts.cluster_latent_relevances
```
"""

import os

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
n_clusters = 15
batch_size = 500
save_files = False
model_name = "64x6-2018_0627_1913_08_161.onnx"
dataset_name = "TCEC_game_collection_random_boards_bestlegal_knight.jsonl"
only_config_rel = True
best_legal = True
run_name = (
    f"bestres_tcec_bestlegal_knight_{'expbest' if best_legal else 'full'}"
)
n_samples = 1000
conv_sum_dims = ()
cosine_sim = False
kmeans_on_tsne = True
viz_latent = True
viz_name = (
    f"{'latent' if viz_latent else 'input'}"
    f"_nosum_{'cosine' if cosine_sim else 'norm'}"
    f"_{'after' if kmeans_on_tsne else 'before'}-tsne"
)
#######################################


def legal_init_rel(board_list, out_tensor):
    legal_move_mask = torch.zeros((len(board_list), 1858))
    for idx, board in enumerate(board_list):
        legal_moves = [
            move_utils.encode_move(move, (board.turn, not board.turn))
            for move in board.legal_moves
        ]
        legal_move_mask[idx, legal_moves] = 1
    return legal_move_mask * out_tensor


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

        def init_rel_fn(out_tensor):
            rel = torch.zeros_like(out_tensor)
            for i in range(rel.shape[0]):
                rel[i, label_tensor[i]] = out_tensor[i, label_tensor[i]]
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
    relevances = relevances[:n_samples]
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

            doc = Document(
                geometry_options={
                    "lmargin": "3cm",
                    "tmargin": "0.5cm",
                    "bmargin": "1.5cm",
                    "rmargin": "3cm",
                }
            )
            doc.packages.append(Package("xskak"))

            # compute heatmap for each nearest neighbor
            for idx_sample in nearest_neighbors:
                _, board, label = concept_dataset[idx_sample]
                _, board_tensor, _ = ConceptDataset.collate_fn_tensor(
                    [concept_dataset[idx_sample]]
                )
                label_tensor = torch.tensor([label])

                def init_rel_fn(out_tensor):
                    rel = torch.zeros_like(out_tensor)
                    for i in range(rel.shape[0]):
                        rel[i, label_tensor[i]] = out_tensor[
                            i, label_tensor[i]
                        ]
                    return rel

                move = move_utils.decode_move(
                    label, (board.turn, not board.turn), board
                )
                uci_move = move.uci()

                if viz_latent:
                    latent_rel = all_relevances[layer_name][idx_sample]
                    if not board.turn:
                        latent_rel = latent_rel.flip(1)
                    latent_rel = latent_rel.view(-1, 64)
                    channel_rels = latent_rel.abs().sum(dim=1)
                    c1, c2 = torch.topk(channel_rels, 2).indices
                    heatmap_str_list = [
                        create_heatmap_string(latent_rel.sum(0), abs_max=True),
                        create_heatmap_string(latent_rel[c1], abs_max=True),
                        create_heatmap_string(latent_rel[c2], abs_max=True),
                    ]
                    heatmap_caption_list = [
                        "Total relevance",
                        "Best channel",
                        "Second best channel",
                    ]
                    add_caption = "latent"
                else:
                    board_tensor.requires_grad = True
                    attr = attribution(
                        board_tensor,
                        [{"y": None}],
                        composite,
                        init_rel=init_rel_fn if best_legal else None,
                    )
                    input_relevances = board_tensor.grad
                    if not board.turn:
                        input_relevances = (
                            input_relevances.view(112, 8, 8)
                            .flip(1)
                            .view(112, 64)
                        )
                    input_relevances = input_relevances.view(112, 64)
                    heatmap_str_list = [
                        create_heatmap_string(
                            input_relevances.sum(dim=0), abs_max=True
                        ),
                        create_heatmap_string(
                            input_relevances[:13].sum(dim=0), abs_max=True
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
                    add_caption = (
                        f"{h0/total:.0f}%|{hist/total:.0f}%|{meta/total:.0f}%"
                    )

                doc = add_plot(
                    doc,
                    board.fen(),
                    heatmap_str_list,
                    current_piece_pos=uci_move[:2],
                    next_move=uci_move[2:4],
                    caption=f"Sample {idx_sample} - {add_caption}",
                    heatmap_caption_list=heatmap_caption_list,
                )

            # Generate pdf
            doc.generate_pdf(
                f"scripts/results/{run_name}"
                f"/{viz_name}/{layer_name.replace('/','.')}"
                f"_cluster_{idx_cluster}",
                clean_tex=True,
            )
