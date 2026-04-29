---
name: tdhook
description: Provides guidance for interpreting and manipulating neural network internals using tdhook with TensorDict and PyTorch hooks. Use when needing attribution maps, activation analysis, probing, steering, activation patching, or weight-level interventions on PyTorch or TensorDict models.
version: 1.0.0
author: Xmaster6y
license: MIT
tags: [tdhook, Interpretability, Attribution, Activation Analysis, Probing, Steering, TensorDict, PyTorch Hooks, GradCAM, LRP, Activation Patching]
dependencies: [tdhook, tensordict>=0.3.0, torch>=2.0.0]
---

# tdhook

Interpretability with TensorDict and PyTorch hooks. One pattern: `with Method(...).prepare(model) as hooked: td = hooked(td)`.

**Docs**: [Home](https://tdhook.readthedocs.io/en/latest/) · [Methods](https://tdhook.readthedocs.io/en/latest/methods.html) · [Tutorials](https://tdhook.readthedocs.io/en/latest/tutorials.html) · [API](https://tdhook.readthedocs.io/en/latest/api/index.html) · **GitHub**: [Xmaster6y/tdhook](https://github.com/Xmaster6y/tdhook) · **Paper**: [arXiv:2509.25475](https://arxiv.org/abs/2509.25475)

## When to Use

**Use tdhook when you need to:**
- Compute input attributions (Saliency, IntegratedGradients, GradCAM, LRP)
- Capture or patch activations at arbitrary layers
- Train probes (linear, bilinear) on representations
- Steer model behavior (ActivationAddition, SteeringVectors)
- Apply weight-level changes (Pruning, Adapters, TaskVectors)
- Work with TensorDictModule, TorchRL, or HuggingFace models

**Consider alternatives:** nnsight (remote 70B+), pyvene (declarative configs), TransformerLens (cached activations).

---

## Workflow 1: Input Attribution (IntegratedGradients)

**Goal**: Which inputs (pixels, tokens) matter most for a prediction?

**Checklist**:
- [ ] Define `init_attr_targets(td, ctx)` returning TensorDict with target outputs
- [ ] Create baseline TensorDict (zeros, blurred, or neutral)
- [ ] Use `TensorDict({"input": x, ("baseline", "input"): baseline})`
- [ ] Call `hooked(td)`; read `td.get(("attr", "input"))`
- [ ] Visualize: heatmap for images, token-level for text

```python
from tdhook.attribution import IntegratedGradients

def init_attr_targets(targets, _):
    return TensorDict(out=targets["output"][..., class_idx], batch_size=targets.batch_size)

with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x, ("baseline", "input"): baseline}))
    attr = td.get(("attr", "input"))
```

---

## Workflow 2: Linear Probing on Layer Activations

**Goal**: Train probes on representations for classification/diagnostics.

**Checklist**:
- [ ] Create `ProbeManager(sklearn_model, kwargs, compute_metrics)`
- [ ] Add `labels` and `step_type` ("fit" / "predict") to TensorDict
- [ ] Choose layer regex: e.g. `"transformer.h.(0|5|10).mlp$"`
- [ ] Pass `additional_keys=["labels", "step_type"]` to Probing
- [ ] For HuggingFace: `in_keys=["input_ids"]`, `out_keys=["logits"]`
- [ ] Call `hooked(train_td)` then `hooked(test_td)`; read `manager.fit_metrics`, `manager.predict_metrics`

```python
from tdhook.latent.probing import Probing, ProbeManager

manager = ProbeManager(LogisticRegression, {}, compute_metrics)
with Probing(
    "transformer.h.(0|5|10).mlp$",
    manager.probe_factory,
    additional_keys=["labels", "step_type"],
).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(train_td)  # step_type="fit"
    hooked(test_td)   # step_type="predict"
```

---

## Workflow 3: Capture or Override Activations (Low-Level)

**Goal**: Inspect or patch activations at specific modules without high-level methods.

**Checklist**:
- [ ] Use `hooked_module.run(data, grad_enabled=...)` for low-level control
- [ ] Inside context: `run.save("path.to.module")` to capture
- [ ] Use `run.set("path.to.module", tensor)` to override
- [ ] Call `proxy.resolve()` after the run to read cached tensors
- [ ] For gradients: `run.save_grad(...)`, `run.set_grad(...)`

```python
with hooked_module.run(data, grad_enabled=True) as run:
    run.save("layers.5.mlp")
    run.set("layers.5.attn", override_tensor)
cached = run.get("layers.5.mlp", cache_key="my_key").resolve()
```

---

## Quick Patterns

```python
# Attribution (needs baseline for IG)
with IntegratedGradients(init_attr_targets=init_fn).prepare(model) as hooked:
    td = hooked(TensorDict({"input": x, ("baseline", "input"): baseline}))
    attr = td.get(("attr", "input"))

# Steering: extract and apply
with ActivationAddition(["layer.7.mlp"]).prepare(model) as hooked:
    steer = hooked(TensorDict({("positive","input"): pos, ("negative","input"): neg})).get(("steer","layer.7.mlp"))
with SteeringVectors(modules_to_steer=["layer.7.mlp"], steer_fn=lambda k, o: o + scale*steer).prepare(model) as hooked:
    out = hooked(TensorDict({"input": x}))
```

## Key TensorDict Keys

| Key | Purpose |
|-----|---------|
| `("baseline", "input")` | Attribution baseline |
| `("positive", "input")`, `("negative", "input")` | Steering extraction |
| `("patched", "input")` | Patching source |
| `("attr", key)` | Attribution output |
| `labels`, `step_type` | Probing (via additional_keys) |

---

## Common Issues & Troubleshooting

| Issue | Solution |
|-------|----------|
| `KeyError` on TensorDict | Use tuple keys: `("attr", "input")`, `("baseline", "input")` |
| HuggingFace model fails | Pass `in_keys=["input_ids"]`, `out_keys=["logits"]` to `prepare()` |
| `BilinearProbeManager` shape mismatch | Call `manager.before_all()` before forwards, `manager.after_all()` after |
| Module path not found | Use `relative=False` or correct regex; see [api.md](references/api.md) Module Path Resolution |
| Probing `step_type` missing | Add `additional_keys=["labels", "step_type"]` and pass both in TensorDict |
| IG baseline wrong shape | Ensure `("baseline", "input")` matches `"input"` shape and device |

See [references/issues.md](references/issues.md) for more patterns.

---

## Feedback Loops for Quality-Critical Operations

- **Attribution**: Validate with Infidelity/Sensitivity metrics (`tdhook.metrics`) before trusting heatmaps
- **Probing**: Compare train vs test metrics to detect overfitting; use cross-validation for probe selection
- **Steering**: Ablation: remove steer at different layers to verify effect

---

## Method Selection Guide

| Need | Primary Class | Key Params |
|------|---------------|------------|
| Gradient w.r.t. input | `Saliency` | `init_attr_targets`, `input_modules` |
| Path-integral attribution | `IntegratedGradients` | `init_attr_targets`, `n_steps`, baseline in TensorDict |
| Channel-weighted spatial | `GradCAM` | `modules_to_attribute` (path → DimsConfig) |
| Extract contrast vector | `ActivationAddition` | module list, `("positive","input")`, `("negative","input")` |
| Apply precomputed vector | `SteeringVectors` | `steer_fn(module_key, output)` |
| Replace activations | `ActivationPatching` | `patch_fn`, `("patched","input")` |
| Train classifiers on reps | `Probing` | `key_pattern`, `probe_factory`, `additional_keys` |
| Zero params by importance | `Pruning` | `importance_callback`, `amount_to_prune` |
| Insert modules inline | `Adapters` | `adapters={path: (adapter, source, target)}` |

---

## Setup & Installation

```bash
pip install tdhook tensordict torch
```

For optional captum-based attribution or sklearn probing:
```bash
pip install captum scikit-learn
```

Colab dev setup: see [tutorials.md](references/tutorials.md) Setup section.

---

## Pitfalls to Avoid

- **IG without baseline**: IntegratedGradients requires `("baseline", "input")`; use zeros or neutral input
- **Probing without step_type**: Probing needs `step_type` in TensorDict ("fit" for training, "predict" for eval)
- **Nested keys as strings**: Use `("attr", "input")` not `"attr/input"` for TensorDict
- **Wrong module path**: Use regex that matches actual submodule names; `transformer.h.5.mlp` vs `layers.5.mlp` depends on model

---

## References

| File | Contents |
|------|----------|
| [references/README.md](references/README.md) | Overview |
| [references/api.md](references/api.md) | Full API: HookedModule, methods by category |
| [references/tutorials.md](references/tutorials.md) | Use-case tutorials |
| [references/issues.md](references/issues.md) | GitHub issues & solutions |
| [references/file_structure.md](references/file_structure.md) | Codebase navigation |

**Official docs**: [Home](https://tdhook.readthedocs.io/en/latest/) · [Methods](https://tdhook.readthedocs.io/en/latest/methods.html) · [Tutorials](https://tdhook.readthedocs.io/en/latest/tutorials.html) · [API Reference](https://tdhook.readthedocs.io/en/latest/api/index.html) · [Releases](https://github.com/Xmaster6y/tdhook/releases)
