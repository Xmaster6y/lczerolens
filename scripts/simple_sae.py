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
import wandb
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.metrics import r2_score

from lczerolens import BoardDataset, ModelWrapper
from lczerolens.xai import ActivationLens, PatchingLens

from .sae_training import trainSAE
from .secret import WANDB_API_KEY

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
parser.add_argument("--sae_module_name", type=str, default="block5/conv2/relu")
parser.add_argument("--dict_size_scale", type=int, default=16)
parser.add_argument("--h_patch_size", type=int, default=1)
parser.add_argument("--w_patch_size", type=int, default=1)
parser.add_argument(
    "--make_symetric_patch",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--sparsity_penalty", type=float, default=1e-2)
parser.add_argument("--train_batch_size", type=int, default=250)
parser.add_argument("--ghost_threshold", type=int, default=4000)
parser.add_argument("--log_steps", type=int, default=50)
parser.add_argument("--warmup_steps", type=int, default=100)
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
wandb.login(key=WANDB_API_KEY)  # type: ignore
wandb.init(  # type: ignore
    project="lczerolens-saes",
    config={
        "model_name": ARGS.model_name,
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
        "train_batch_size": ARGS.train_batch_size,
        "ghost_threshold": ARGS.ghost_threshold,
        "log_steps": ARGS.log_steps,
        "warmup_steps": ARGS.warmup_steps,
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
    with safe_open(
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
        "test_activations.safetensors",
        framework="pt",
    ) as f:
        test_activations = f.get_tensor(ARGS.sae_module_name)

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

    train_activations = rearrange_activations(train_activations)
    val_activations = rearrange_activations(val_activations)
    test_activations = rearrange_activations(test_activations)

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

    act_dim = train_activations.shape[1]
    ae = trainSAE(
        train_dataloader,
        act_dim,
        ARGS.dict_size_scale * act_dim,
        val_dataloader=val_dataloader,
        sparsity_penalty=ARGS.sparsity_penalty,
        lr=ARGS.lr,
        beta1=ARGS.beta1,
        beta2=ARGS.beta2,
        log_steps=ARGS.log_steps,
        val_steps=ARGS.val_steps,
        warmup_steps=ARGS.warmup_steps,
        n_epochs=ARGS.n_epochs,
        ghost_threshold=ARGS.ghost_threshold,
        resample_steps=None,
        device=DEVICE,
        from_checkpoint=ARGS.from_checkpoint,
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
    ae = torch.load(
        f"{ARGS.output_root}/scripts/saes/{ARGS.model_name}/"
        f"{ARGS.sae_module_name.replace('/', '_')}.pt"
    )
    ae.to(DEVICE)

    def sae_patch(x, **kwargs):
        x_c_batched = rearrange_activations(x)
        act_c_batched = ae(x_c_batched)["x_hat"].detach()
        act_batched = invert_rearrange_activations(act_c_batched)
        return act_batched

    patching_lens = PatchingLens(
        {
            ARGS.sae_module_name: sae_patch,
        }
    )
    patched_batched_outs = patching_lens.analyse_dataset(
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
    if "value" in batched_outs.keys():
        value_r2 = r2_score(
            batched_outs["value"].cpu(),
            patched_batched_outs["value"].cpu(),
        )
        wandb.log({"test/value_r2": value_r2})  # type: ignore
    if "policy" in batched_outs.keys():
        policy_r2 = r2_score(
            batched_outs["policy"].cpu(),
            patched_batched_outs["policy"].cpu(),
        )
        wandb.log({"test/policy_r2": policy_r2})  # type: ignore
    if "wdl" in batched_outs.keys():
        wdl_bc = torch.nn.functional.binary_cross_entropy(
            batched_outs["wdl"].cpu(),
            patched_batched_outs["wdl"].cpu(),
        )
        wandb.log({"test/wdl_bc": wdl_bc})  # type: ignore
    if "mlh" in batched_outs.keys():
        mlh_r2 = r2_score(
            batched_outs["mlh"].cpu(), patched_batched_outs["mlh"].cpu()
        )
        wandb.log({"test/mlh_r2": mlh_r2})  # type: ignore
