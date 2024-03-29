"""Train a simple autoencoder.

Run with:
```bash
poetry run python -m scripts.simple_sae
```

Wandb:
```yaml
command:
  - poetry
  - run
  - python
  - -m
  - ${program}
  - ${args}
method: bayes
metric:
  goal: maximize
  name: val/r2_score
parameters:
  model_name:
    value: maia-1100.onnx
  sae_module_name:
    value: block5/conv2/relu
#   from_checkpoint:
#     value: null
  freeze_dict:
    value: false
  beta1:
    distribution: inv_log_uniform_values
    max: 1
    min: 0.95
    # value: 0.99
  beta2:
    distribution: inv_log_uniform_values
    max: 1
    min: 0.995
    # value: 0.999
  dict_size_scale:
    distribution: int_uniform
    max: 126
    min: 8
    # value: 50
  ghost_threshold:
    distribution: int_uniform
    max: 8000
    min: 100
    # value: 4000
  resample_steps:
    distribution: int_uniform
    max: 8000
    min: 100
    # value: 4000
  lr:
    distribution: log_uniform_values
    max: 1e-3
    min: 1e-5
  n_epochs:
    distribution: int_uniform
    max: 60
    min: 15
  h_patch_size:
    values: [1, 2, 4]
    # value: 4
  make_symetric_patch:
    values: [true, false]
    # value: true
  w_patch_size:
    values: [1, 2, 4]
    # value: 4
  sparsity_penalty:
    distribution: log_uniform_values
    max: 1
    min: 5e-3
  less_than_1:
    values: [true, false]
  train_batch_size:
    value: 250
  warmup_steps:
    distribution: int_uniform
    max: 200
    min: 10
    # value: 50
  cooldown_steps:
    distribution: int_uniform
    max: 400
    min: 10
    # value: 150
program: scripts.simple_sae
```
"""

import argparse
import os

import einops
import numpy as np
import torch
import wandb
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    r2_score,
)

from lczerolens import BoardDataset, ModelWrapper
from lczerolens.xai import ActivationLens, PatchingLens

from .sae_training import trainSAE

#######################################
# HYPERPARAMETERS
#######################################
parser = argparse.ArgumentParser("simple-sae")
parser.add_argument("--output-root", type=str, default=".")
# Activation lens
parser.add_argument(
    "--compte_activations",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--act_batch_size", type=int, default=100)
parser.add_argument("--model_name", type=str, default="maia-1100.onnx")
# SAE training
parser.add_argument(
    "--train_sae", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--from_checkpoint", type=str, default=None)
parser.add_argument("--freeze_dict", type=bool, default=False)
parser.add_argument("--sae_module_name", type=str, default="block1/conv2/relu")
parser.add_argument("--dict_size_scale", type=int, default=16)
parser.add_argument("--h_patch_size", type=int, default=1)
parser.add_argument("--w_patch_size", type=int, default=1)
parser.add_argument(
    "--make_symetric_patch",
    type=bool,
    default=True,
)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--sparsity_penalty", type=float, default=1e-2)
parser.add_argument("--less_than_1", type=bool, default=True)
parser.add_argument("--ghost_threshold", type=int, default=4000)
parser.add_argument("--resample_steps", type=int, default=4000)
parser.add_argument("--train_batch_size", type=int, default=250)
parser.add_argument("--eval_batch_size", type=int, default=500)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--cooldown_steps", type=int, default=200)
parser.add_argument("--log_steps", type=int, default=50)
parser.add_argument("--val_steps", type=int, default=200)
# Test
parser.add_argument(
    "--compute_evals", action=argparse.BooleanOptionalAction, default=True
)
#######################################

ARGS = parser.parse_args()
if ARGS.make_symetric_patch:
    ARGS.h_patch_size = 4
    ARGS.w_patch_size = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

run_name = f"{ARGS.model_name}_{ARGS.sae_module_name.replace('/', '_')}"
os.makedirs(f"{ARGS.output_root}/ignored/saes", exist_ok=True)
wandb.login()  # type: ignore
wandb.init(  # type: ignore
    project="lczerolens-saes",
    config={
        "model_name": ARGS.model_name,
        "from_checkpoint": ARGS.from_checkpoint,
        "freeze_dict": ARGS.freeze_dict,
        "sae_module_name": ARGS.sae_module_name,
        "dict_size_scale": ARGS.dict_size_scale,
        "h_patch_size": ARGS.h_patch_size,
        "w_patch_size": ARGS.w_patch_size,
        "make_symetric_patch": ARGS.make_symetric_patch,
        "lr": ARGS.lr,
        "beta1": ARGS.beta1,
        "beta2": ARGS.beta2,
        "n_epochs": ARGS.n_epochs,
        "sparsity_penalty": ARGS.sparsity_penalty,
        "less_than_1": ARGS.less_than_1,
        "ghost_threshold": ARGS.ghost_threshold,
        "resample_steps": ARGS.resample_steps,
        "train_batch_size": ARGS.train_batch_size,
        "eval_batch_size": ARGS.eval_batch_size,
        "warmup_steps": ARGS.warmup_steps,
        "cooldown_steps": ARGS.cooldown_steps,
        "log_steps": ARGS.log_steps,
        "val_steps": ARGS.val_steps,
    },
)

print(f"[INFO] Running on {DEVICE}")
model = ModelWrapper.from_path(f"./assets/{ARGS.model_name}")
model.to(DEVICE)
base_dataset_name = "TCEC_game_collection_random_boards"
train_artifacts = wandb.use_artifact("tcec_train:latest")  # type: ignore
train_dataset = BoardDataset(f"./assets/{base_dataset_name}_train.jsonl")
val_artifacts = wandb.use_artifact("tcec_val:latest")  # type: ignore
val_dataset = BoardDataset(f"./assets/{base_dataset_name}_val.jsonl")
test_artifacts = wandb.use_artifact("tcec_test:latest")  # type: ignore
test_dataset = BoardDataset(f"./assets/{base_dataset_name}_test.jsonl")

if ARGS.compte_activations:
    for dataset_type, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ]:
        activation_lens = ActivationLens(module_exp=r"block\d+/conv2/relu")
        activations = activation_lens.analyse_dataset(
            dataset,
            model,
            batch_size=ARGS.act_batch_size,
            collate_fn=BoardDataset.collate_fn_tuple,
        )

        os.makedirs(
            f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}", exist_ok=True
        )
        save_file(
            activations,
            f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
            f"{dataset_type}_activations.safetensors",
        )


def collate_fn(b):
    (acts,) = zip(*b)
    return torch.stack(acts)


if ARGS.make_symetric_patch:

    def rearrange_activations(activations):
        p1 = activations[:, :, :4, :4]
        p2 = activations[:, :, :4, 4:].flip(dims=(3,))
        p3 = activations[:, :, 4:, :4].flip(dims=(2,))
        p4 = activations[:, :, 4:, 4:].flip(dims=(2, 3))
        patches = torch.cat([p1, p2, p3, p4], dim=1)
        return einops.rearrange(patches, "b c h w -> (b h w) c")

    def invert_rearrange_activations(activations):
        patches = einops.rearrange(
            activations, "(b h w) c -> b c h w", h=4, w=4
        )
        p1 = patches[:, :64]
        p2 = patches[:, 64:128].flip(dims=(3,))
        p3 = patches[:, 128:192].flip(dims=(2,))
        p4 = patches[:, 192:].flip(dims=(2, 3))
        return torch.cat(
            [
                torch.cat([p1, p2], dim=3),
                torch.cat([p3, p4], dim=3),
            ],
            dim=2,
        )

else:

    def rearrange_activations(activations):
        split_batch = einops.rearrange(
            activations,
            "b c (h ph) (w pw) -> b c h ph w pw",
            ph=ARGS.h_patch_size,
            pw=ARGS.w_patch_size,
        )
        return einops.rearrange(
            split_batch, "b c h ph w pw -> (b h w) (c ph pw)"
        )

    def invert_rearrange_activations(activations):
        split_batch = einops.rearrange(
            activations,
            "(b h w) (c ph pw) -> b c h ph w pw",
            h=8 // ARGS.h_patch_size,
            w=8 // ARGS.w_patch_size,
            ph=ARGS.h_patch_size,
            pw=ARGS.w_patch_size,
        )
        return einops.rearrange(
            split_batch, "b c h ph w pw -> b c (h ph) (w pw)"
        )


if ARGS.train_sae:
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
        "train_activations.safetensors",
        framework="pt",
    ) as f:
        train_activations = f.get_tensor(ARGS.sae_module_name)
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
        "val_activations.safetensors",
        framework="pt",
    ) as f:
        val_activations = f.get_tensor(ARGS.sae_module_name)

    train_activations = rearrange_activations(train_activations)
    val_activations = rearrange_activations(val_activations)

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            train_activations,
        ),
        batch_size=ARGS.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            val_activations,
        ),
        batch_size=ARGS.eval_batch_size,
        collate_fn=collate_fn,
    )

    act_dim = train_activations.shape[1]
    ae = trainSAE(
        train_dataloader,
        act_dim,
        ARGS.dict_size_scale * act_dim,
        val_dataloader=val_dataloader,
        sparsity_penalty=ARGS.sparsity_penalty,
        less_than_1=ARGS.less_than_1,
        lr=ARGS.lr,
        beta1=ARGS.beta1,
        beta2=ARGS.beta2,
        log_steps=ARGS.log_steps,
        val_steps=ARGS.val_steps,
        warmup_steps=ARGS.warmup_steps,
        n_epochs=ARGS.n_epochs,
        ghost_threshold=ARGS.ghost_threshold,
        resample_steps=ARGS.resample_steps,
        device=DEVICE,
        from_checkpoint=ARGS.from_checkpoint,
        freeze_dict=ARGS.freeze_dict,
        wandb=wandb,
    )
    model_path = (
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}"
        f"/{ARGS.sae_module_name.replace('/', '_')}.pt"
    )

    torch.save(
        ae,
        model_path,
    )
    print(f"[INFO] Model saved to {model_path}")

    artifact = wandb.Artifact(  # type: ignore
        f"{run_name}.pt",
        type="model",
    )
    artifact.add_file(
        model_path,
    )
    wandb.log_artifact(artifact)  # type: ignore

if ARGS.compute_evals:
    if not ARGS.train_sae:
        ae = torch.load(
            f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
            f"{ARGS.sae_module_name.replace('/', '_')}.pt",
            map_location=torch.device(DEVICE),
        )
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
        "test_activations.safetensors",
        framework="pt",
    ) as f:
        test_activations = f.get_tensor(ARGS.sae_module_name)
    test_activations = rearrange_activations(test_activations)

    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            test_activations,
        ),
        batch_size=ARGS.eval_batch_size,
        collate_fn=collate_fn,
    )
    test_losses = {
        "explained_variance": 0.0,
        "r2_score": 0.0,
    }
    feature_act_count = torch.zeros(ae.dict_size)
    activated_features = 0
    with torch.no_grad():
        for acts in test_dataloader:
            out = ae(acts.to(DEVICE), output_features=True)
            f = out["features"]
            x_hat = out["x_hat"]
            test_losses["explained_variance"] += explained_variance_score(
                acts.cpu(), x_hat.cpu()
            )
            test_losses["r2_score"] += r2_score(acts.cpu(), x_hat.cpu())
            feature_act_count += (f > 0).sum(dim=0).cpu()
            activated_features += (f > 0).sum().cpu()
        hist = np.histogram(
            feature_act_count.numpy() / len(test_dataset),
            density=True,
            bins=20,
        )
        wandb.log(  # type: ignore
            {
                "test/feature_density": wandb.Histogram(  # type: ignore
                    np_histogram=hist
                )
            }
        )
        wandb.log(  # type: ignore
            {
                "test/ativated_features": activated_features
                / len(test_dataset),
                "test/frac_activated_features": activated_features
                / ae.dict_size
                / len(test_dataset),
            }
        )
        for k in test_losses.keys():
            test_losses[k] /= len(test_dataloader)
            wandb.log({f"test/{k}": test_losses[k]})  # type: ignore

    def sae_patch(x, **kwargs):
        x_c_batched = rearrange_activations(x)
        act_c_batched = ae(x_c_batched)["x_hat"].detach()
        act_batched = invert_rearrange_activations(act_c_batched)
        return act_batched

    def null_patch(x, **kwargs):
        return torch.zeros_like(x)

    def rand_patch(x, **kwargs):
        return torch.rand_like(x)

    patch_lenses = {
        "sae": PatchingLens(
            {
                ARGS.sae_module_name: sae_patch,
            }
        ),
        "null": PatchingLens(
            {
                ARGS.sae_module_name: null_patch,
            }
        ),
        "rand": PatchingLens(
            {
                ARGS.sae_module_name: rand_patch,
            }
        ),
    }
    patched_batched_outs = {}
    for k, lens in patch_lenses.items():
        patched_batched_outs[k] = lens.analyse_dataset(
            test_dataset,
            model,
            batch_size=ARGS.train_batch_size,
            collate_fn=BoardDataset.collate_fn_tuple,
        )
    identity_patching_lens = PatchingLens({})
    batched_outs = identity_patching_lens.analyse_dataset(
        test_dataset,
        model,
        batch_size=ARGS.train_batch_size,
        collate_fn=BoardDataset.collate_fn_tuple,
    )
    for out_k in batched_outs.keys():
        for patch_name in patched_batched_outs.keys():
            r2 = r2_score(
                batched_outs[out_k].cpu(),
                patched_batched_outs[patch_name][out_k].cpu(),
            )
            wandb.log({f"test/{out_k}_{patch_name}_r2": r2})  # type: ignore
            mse = mean_squared_error(
                batched_outs[out_k].cpu(),
                patched_batched_outs[patch_name][out_k].cpu(),
            )
            wandb.log({f"test/{out_k}_{patch_name}_mse": mse})  # type: ignore

    print("[INFO] evaluation done")
