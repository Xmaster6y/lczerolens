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
program: ignored.simple_sae
metric:
  goal: minimize
  name: val_mse
```
"""

import argparse
import os

import einops
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import wandb
from lczerolens import BoardDataset, ModelWrapper
from lczerolens.xai import ActivationLens

from .sae_training import trainSAE
from .secret import WANDB_API_KEY

#######################################
# HYPERPARAMETERS
#######################################
parser = argparse.ArgumentParser("simple-sae")
parser.add_argument("--output-root", type=str, default=".")
# Activation lens
compte_activations = False
act_batch_size = 100
model_name = "maia-1100.onnx"
# SAE training
train_sae = True
sae_module_name = "block5/conv2/relu"
run_name = f"{model_name}_{sae_module_name.replace('/', '_')}"
parser.add_argument("--dict_size_scale", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--sparsity_penalty", type=float, default=1e-2)
parser.add_argument("--train_batch_size", type=int, default=256)
parser.add_argument("--ghost_threshold", type=int, default=400)
parser.add_argument("--log_steps", type=int, default=50)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--val_steps", type=int, default=100)
# Test
compute_evals = False
#######################################

ARGS = parser.parse_args()
print(ARGS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{ARGS.output_root}/ignored/saes", exist_ok=True)
wandb.login(key=WANDB_API_KEY)  # type: ignore
wandb.init(  # type: ignore
    project="lczerolens-saes",
    config={
        "model_name": model_name,
        "sae_module_name": sae_module_name,
        "lr": ARGS.lr,
        "n_epochs": ARGS.n_epochs,
        "sparsity_penalty": ARGS.sparsity_penalty,
        "dict_size_scale": ARGS.dict_size_scale,
        "train_batch_size": ARGS.train_batch_size,
        "ghost_threshold": ARGS.ghost_threshold,
        "log_steps": ARGS.log_steps,
        "warmup_steps": ARGS.warmup_steps,
        "val_steps": ARGS.val_steps,
    },
)

print(f"Running on {DEVICE}")
model = ModelWrapper.from_path(f"./assets/{model_name}")
model.to(DEVICE)
base_dataset_name = "TCEC_game_collection_random_boards"
train_artifacts = wandb.use_artifact("tcec_train:latest")  # type: ignore
train_dataset = BoardDataset(f"./assets/{base_dataset_name}_train.jsonl")
val_artifacts = wandb.use_artifact("tcec_val:latest")  # type: ignore
val_dataset = BoardDataset(f"./assets/{base_dataset_name}_val.jsonl")
test_artifacts = wandb.use_artifact("tcec_test:latest")  # type: ignore
test_dataset = BoardDataset(f"./assets/{base_dataset_name}_test.jsonl")

if compte_activations:
    for dataset_type, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ]:
        activation_lens = ActivationLens(module_exp=r"block\d+/conv2/relu")
        activations = activation_lens.analyse_dataset(
            dataset,
            model,
            batch_size=act_batch_size,
            collate_fn=BoardDataset.collate_fn_tuple,
        )

        os.makedirs(
            f"{ARGS.output_root}/scripts/saes/{model_name}", exist_ok=True
        )
        save_file(
            activations,
            f"{ARGS.output_root}/scripts/saes/{model_name}/"
            f"{dataset_type}_activations.safetensors",
        )

if train_sae:
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{model_name}/"
        "train_activations.safetensors",
        framework="pt",
    ) as f:
        train_activations = f.get_tensor(sae_module_name)
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{model_name}/"
        "val_activations.safetensors",
        framework="pt",
    ) as f:
        val_activations = f.get_tensor(sae_module_name)
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{model_name}/"
        "test_activations.safetensors",
        framework="pt",
    ) as f:
        test_activations = f.get_tensor(sae_module_name)

    train_activations = einops.rearrange(
        train_activations, "b c h w -> (b h w) c"
    )
    val_activations = einops.rearrange(val_activations, "b c h w -> (b h w) c")
    test_activations = einops.rearrange(
        test_activations, "b c h w -> (b h w) c"
    )

    def collate_fn(b):
        (acts,) = zip(*b)
        return torch.stack(acts)

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
        batch_size=ARGS.train_batch_size,
        collate_fn=collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            test_activations,
        ),
        batch_size=ARGS.train_batch_size,
        collate_fn=collate_fn,
    )

    act_dim = train_activations.shape[1]
    ae = trainSAE(
        train_dataloader,
        act_dim,
        ARGS.dict_size_scale * act_dim,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        sparsity_penalty=ARGS.sparsity_penalty,
        lr=ARGS.lr,
        log_steps=ARGS.log_steps,
        val_steps=ARGS.val_steps,
        warmup_steps=ARGS.warmup_steps,
        n_epochs=ARGS.n_epochs,
        ghost_threshold=ARGS.ghost_threshold,
        resample_steps=None,
        device=DEVICE,
        wandb=wandb,
    )
    torch.save(
        ae,
        f"{ARGS.output_root}/scripts/saes/{run_name}/"
        f"{sae_module_name.replace('/', '_')}_{ARGS.dict_size_scale}.pt",
    )
