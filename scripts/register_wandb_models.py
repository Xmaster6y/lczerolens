"""Register models in Weights & Biases.

Run with:
```bash
poetry run python -m scripts.register_wandb_models
```
"""

import argparse

import wandb

#######################################
# HYPERPARAMETERS
#######################################
parser = argparse.ArgumentParser("register-wandb-models")
models = {
    "maia-1100": "maia-1100.onnx",
}
parser.add_argument(
    "--log_models", action=argparse.BooleanOptionalAction, default=False
)
#######################################

ARGS = parser.parse_args()

if ARGS.log_models:
    wandb.login()  # type: ignore
    with wandb.init(  # type: ignore
        project="lczerolens-saes", job_type="make-models"
    ) as run:
        for model_name, model_path in models.items():
            artifact = wandb.Artifact(model_name, type="model")  # type: ignore
            artifact.add_file(f"./assets/{model_path}")
            run.log_artifact(artifact)
