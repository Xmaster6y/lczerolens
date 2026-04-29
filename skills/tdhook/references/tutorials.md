# tdhook Tutorials

Specific use cases with step-by-step workflows.

---

## Use Case: Image Class Attribution (VGG16)

**Goal**: Which pixels matter most for a class prediction?

```python
import torch, timm
from tensordict import TensorDict
from tdhook.attribution import IntegratedGradients

model = timm.create_model("vgg16.tv_in1k", pretrained=True)
transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
image_tensor = transforms(image)  # [C, H, W]

def init_attr_targets(targets, _):
    return TensorDict(out=targets["output"][..., 340], batch_size=targets.batch_size)  # zebra = 340

with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = TensorDict({
        "input": image_tensor,
        ("baseline", "input"): torch.zeros_like(image_tensor),
    }).unsqueeze(0)
    td = hooked(td)
    attr = td.get(("attr", "input"))
# Visualise attr as heatmap
```

---

## Use Case: Sentiment Probing on GPT-2

**Goal**: Train linear probes on layer activations for sentiment classification (IMDB).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tdhook.latent.probing import Probing, ProbeManager
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def compute_metrics(preds, labels):
    return {"accuracy": accuracy_score(labels, preds)}

manager = ProbeManager(LogisticRegression, {}, compute_metrics)

with Probing(
    "transformer.h.(0|5|10).mlp$",
    manager.probe_factory,
    additional_keys=["labels", "step_type"],
).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    with torch.no_grad():
        hooked(TensorDict({"input_ids": train_ids, "labels": train_labels, "step_type": "fit"}, batch_size=...))
        hooked(TensorDict({"input_ids": test_ids, "labels": test_labels, "step_type": "predict"}, batch_size=...))

print("Train:", manager.fit_metrics, "Test:", manager.predict_metrics)
```

---

## Use Case: Steering Vector (Rich vs Poor)

**Goal**: Extract a steering vector from contrasting prompts and apply it.

```python
from tensordict import TensorDict
from tdhook.latent import ActivationAddition, SteeringVectors

positive_inputs = tokenizer.encode("I am rich.", return_tensors="pt")
negative_inputs = tokenizer.encode("I am poor.", return_tensors="pt")

# 1. Extract
with ActivationAddition(["transformer.h.7.mlp"]).prepare(model) as hooked:
    td = hooked(TensorDict({
        ("positive", "input"): positive_inputs,
        ("negative", "input"): negative_inputs,
    }, batch_size=1))
    steering_vector = td.get(("steer", "transformer.h.7.mlp")).sum(dim=0)

# 2. Apply
def steer_fn(module_key, output):
    return output + scale * steering_vector
with SteeringVectors(modules_to_steer=["transformer.h.7.mlp"], steer_fn=steer_fn).prepare(model) as hooked:
    hooked(TensorDict({"input": base_inputs}, batch_size=1))
```

---

## Use Case: Bilinear Probing (Layer Pairs)

**Goal**: Train bilinear probes on pairs of layers (e.g. 0 and 5) for classification.

```python
from tdhook.latent.probing import Probing, BilinearProbeManager, LowRankBilinearEstimator

def preprocess_last_token(data):
    data = data.detach()
    if data.dim() > 2:
        data = data[:, -1, :]
    return data.flatten(1)

manager = BilinearProbeManager(
    pairs=[("transformer.h.0", "transformer.h.5")],
    estimator_class=LowRankBilinearEstimator,
    estimator_kwargs={"d_latent1": 768, "d_latent2": 768, "num_classes": 2, "epochs": 100, ...},
    compute_metrics=compute_metrics,
    data_preprocess_callback=preprocess_last_token,
    allow_overwrite=True,
)

manager.before_all()
with Probing(manager.key_pattern, manager.probe_factory, additional_keys=["labels", "step_type"], relative=False).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(train_td)
    hooked(test_td)
manager.after_all()
```

---

## Use Case: Intrinsic Dimension of Activations

**Goal**: Estimate intrinsic dimension of activation manifolds (synthetic or from model).

```python
from tdhook.latent.dimension_estimation import TwoNnDimensionEstimator, LocalKnnDimensionEstimator

# Synthetic 2D plane in 10D
plane_data = torch.randn(200, 10)
plane_data[:, 2:] = 0
td = TensorDict({"data": plane_data}, batch_size=[])

twonn = TwoNnDimensionEstimator(return_xy=True)
td_out = twonn(td.clone())
print("TwoNN dim:", td_out["dimension"].item())

est = LocalKnnDimensionEstimator(k="auto")
td_est = est(td.clone())
print("LocalKnn dims:", td_est["dimension"])
```

---

## Use Case: PPO Actor Action Probing

**Goal**: Predict actions from actor MLP activations (TorchRL PPO).

```python
from tdhook.latent.probing import Probing, ProbeManager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def compute_metrics(preds, labels):
    return {"r2": r2_score(labels, preds)}

manager = ProbeManager(LinearRegression, {}, compute_metrics)
with Probing("module.0$", manager.probe_factory, additional_keys=["labels", "step_type"]).prepare(
    actor, in_keys=actor.in_keys, out_keys=actor.out_keys
) as hooked:
    batch["labels"] = batch["action"]
    batch["step_type"] = "fit"
    hooked(batch)
```

---

## Use Case: Chess Board Saliency

**Goal**: Which squares matter for the model's best-move prediction?

```python
from lczerolens import LczeroModel, LczeroBoard
from tensordict import TensorDict
from tdhook.attribution import Saliency

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard(fen)
for move in moves.split(" "):
    board.push_uci(move)
td = model(board)

def best_logit_init_targets(td, _):
    return TensorDict(out=td["policy"].max(dim=-1).values, batch_size=td.batch_size)

with Saliency(init_attr_targets=best_logit_init_targets).prepare(model) as hooked:
    td = hooked(td)
    attr = td.get(("attr", "input"))
# Visualise on board
```

---

## Use Case: Activation Patching

**Goal**: Replace activations at a module with those from another input to test counterfactual effect.

```python
from tensordict import TensorDict
from tdhook.latent.activation_patching import ActivationPatching

def patch_fn(output, output_to_patch, **_):
    output[:, 0] = output_to_patch[:, 0]
    return output

with ActivationPatching(["linear2"], patch_fn=patch_fn).prepare(model) as hooked:
    data = TensorDict({
        "input": source_input,
        ("patched", "input"): patched_input,
    }, batch_size=2)
    data = hooked(data)
    patched_output = data.get(("patched", "output"))
```

---

## Use Case: Weight Pruning by Importance

**Goal**: Prune 50% of parameters by magnitude, run forward, restore on exit.

```python
from tdhook.weights.pruning import Pruning

def importance_cb(parameter, **_):
    return parameter.abs()

pruning = Pruning(importance_callback=importance_cb, amount_to_prune=0.5)
with pruning.prepare(model) as hooked:
    output = hooked(inp)  # Pruned
# Model restored
```

---

## Use Case: Inline Adapters

**Goal**: Insert a custom module (e.g. scaling) at a layer during forward.

```python
from torch import nn
from tdhook.weights.adapters import Adapters

class DoubleAdapter(nn.Module):
    def forward(self, x, **_):
        return x * 2

adapters = {"linear2": (DoubleAdapter(), "linear2", "linear2")}
with Adapters(adapters=adapters).prepare(model) as hooked:
    patched_out = hooked(data)["output"]
```

---

## Use Case: Task Vector Merging

**Goal**: Combine pretrained and finetuned weights via task vectors; pick alpha by validation.

```python
from tdhook.weights.task_vectors import TaskVectors

def get_test_accuracy(module):
    return 0.8  # e.g. eval on holdout

def get_control_adequacy(module):
    return True

task_vectors = TaskVectors(
    alphas=[0.1, 0.5, 1.0],
    get_test_accuracy=get_test_accuracy,
    get_control_adequacy=get_control_adequacy,
)
with task_vectors.prepare(pretrained) as hooked:
    vector = hooked.get_task_vector(finetuned)
    alpha = hooked.hooking_context.compute_alpha(vector)
    new_weights = hooked.get_weights(vector, forget_vector, alpha=alpha)
```

---

## Setup (Colab vs Local)

```python
import importlib.util
MODE = "colab" if importlib.util.find_spec("google.colab") else "local"
if MODE == "colab":
    %pip install -q tdhook
```

---

## Resources

- [Documentation](https://tdhook.readthedocs.io)
- [GitHub](https://github.com/Xmaster6y/tdhook)
- [arXiv paper](https://arxiv.org/abs/2509.25475)
- Notebooks: `docs/source/notebooks/methods/`, `docs/source/notebooks/tutorials/`
