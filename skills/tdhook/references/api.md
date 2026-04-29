# tdhook API Reference

Key classes, usage patterns, and method implementations.

## HookingContextFactory

Base for all method implementations. Subclasses: `IntegratedGradients`, `Saliency`, `Probing`, `ActivationAddition`, etc.

```python
method.prepare(module, in_keys=None, out_keys=None, return_context=True)
# Returns HookingContext (context manager) or HookedModule if return_context=False
```

- `in_keys` / `out_keys`: Override for non-TensorDictModule (e.g. HuggingFace: `["input_ids"]`, `["logits"]`).
- `return_context=False`: Returns `HookedModule` directly; context is auto-entered.

## HookingContext / HookingContextWithCache

Returned by `prepare()` (as context manager). `HookingContextWithCache` adds a `cache` TensorDict and `clear()`; used by methods that cache activations (e.g. probing).

## HookedModule

Wrapper returned inside `with method.prepare(model)`. Callable with TensorDict.

```python
td = hooked_module(td)
# Results written into td
```

### Get / Set / Save API

Use `run(data)` for low-level access. Inside the context, register hooks before the forward executes on exit.

```python
with hooked_module.run(data) as run:
    run.save("layers.5.mlp")           # Capture forward output into run.cache
    run.set("layers.5.attn", override)  # Override activation at that module
    run.get("layers.3.mlp", cache_key="custom")  # Cache with custom key
```

- **`save(key)`** – Capture activation; auto cache key (`key_output` or `key_grad_input`).
- **`get(key, cache_key=...)`** – Same as save but explicit cache key; returns `CacheProxy`.
- **`set(key, value)`** – Replace activation. `value` can be a tensor or `CacheProxy.resolve()`.
- **`stop(key)`** – Raise `EarlyStoppingException` at that module (short-circuit).

Direction-specific variants (all take same args as base):

| Variant | Direction | Use |
|---------|-----------|-----|
| `save` / `get` / `set` | `fwd` | Forward output |
| `save_input` / `get_input` / `set_input` | `fwd_pre` | Forward input |
| `save_grad` / `get_grad` / `set_grad` | `bwd` | Gradient input (requires `grad_enabled=True`) |
| `save_grad_output` / `get_grad_output` / `set_grad_output` | `bwd_pre` | Gradient output |

```python
with hooked_module.run(data, grad_enabled=True) as run:
    run.save_input("layers.0")
    run.save_grad("layers.5.mlp")
```

Common params: `callback=fn`, `prepend=False`, `relative=True`. Paths use module resolution (see Module Path Resolution).

**CacheProxy** – Returned by `get`/`save`. Call `proxy.resolve()` after the run to read the cached tensor.

**Run options**: `run(data, cache=None, run_name="run", run_sep=".", grad_enabled=False, run_callback=None)`. Pass `cache=my_td` to write saves into a shared TensorDict; `run_callback` overrides the default `module(data)` execution.

### Context control

```python
with hooked_module.disable_context_hooks():
    ...  # Run without method hooks

with hooked_module.disable_context() as raw_module:
    ...  # Raw module, no hooks

hooked_module.restore()  # When using prepare(return_context=False)
```

## TensorDict Keys

| Key pattern | Method | Purpose |
|-------------|--------|---------|
| `"input"`, `"output"` | Default | Base I/O for nn.Module |
| `("baseline", "input")` | IntegratedGradients | Baseline for path integral |
| `("attr", "input")` | Attribution | Attribution map |
| `("positive", "input")`, `("negative", "input")` | ActivationAddition | Source prompts |
| `("steer", "module.path")` | ActivationAddition | Steering vector output |
| `"labels"`, `"step_type"` | Probing | Passed via additional_keys |

## Module Path Resolution

Submodule keys in `set`/`get`/`save`/`stop` resolve via `resolve_submodule_path`:

- `layers[0].attention` – indexing
- `layers[-1]`, `layers[1:3]` – slicing
- `<custom/attr>.submodule` – attributes with special chars (e.g. `block0/module`)
- `m1.<0>.layers` – numeric attribute names

Paths default to relative to `td_module`; use `relative=False` for absolute. Probing methods use regex `key_pattern` to match paths (e.g. `"transformer.h.5.mlp$"`).

---

## Method Implementations

High-level modules by category. All extend `HookingContextFactory` and use `prepare(module)`.

### Attribution

Explain which inputs or layers contribute. All write to `("attr", key)`.

| Class | Use |
|-------|-----|
| `Saliency` | Gradient w.r.t. input (or latent). Params: `absolute`, `multiply_by_inputs`, `input_modules`, `target_modules` |
| `IntegratedGradients` | Path integral. Requires `("baseline", "input")`. Params: `n_steps`, `method`, `baseline_key` |
| `GuidedBackpropagation` | ReLU-guided gradients. Params: `input_modules`, `use_inputs` |
| `GradCAM` | Channel-weighted spatial. Params: `modules_to_attribute` (path → `DimsConfig`) |
| `LRP` | Layer-wise Relevance Propagation. Params: `rule_mapper`, `init_attr_grads` |
| `ActivationMaximisation` | PGD to maximise target. Writes to `("attr", "input")` |

```python
from tdhook.attribution import Saliency, IntegratedGradients

with Saliency(init_attr_targets=init_fn).prepare(model) as hooked:
    attr = hooked(TensorDict({"input": x})).get(("attr", "input"))

with IntegratedGradients(init_attr_targets=init_fn).prepare(model) as hooked:
    attr = hooked(TensorDict({"input": x, ("baseline", "input"): baseline})).get(("attr", "input"))
```

### Latent

| Class | Use |
|-------|-----|
| `ActivationAddition` | Extract `positive - negative` at modules. Requires `("positive", "input")`, `("negative", "input")`. Outputs `("steer", module_key)` |
| `SteeringVectors` | Apply `steer_fn(module_key, output)` at modules |
| `ActivationPatching` | Replace activations via `patch_fn(output, output_to_patch, ...)`. Requires `("patched", "input")` |
| `ActivationCaching` | Cache activations at regex-matched modules. `hooked.hooking_context.cache` |
| `Probing` | Train probes via `ProbeManager` / `BilinearProbeManager`. `additional_keys=["labels", "step_type"]` |
| `TwoNnDimensionEstimator`, `LocalKnnDimensionEstimator`, etc. | Intrinsic dimension of `TensorDict({"data": activations})` |

```python
from tdhook.latent import ActivationAddition, SteeringVectors
from tdhook.latent.activation_caching import ActivationCaching
from tdhook.latent.probing import Probing, ProbeManager

# Extract steering vector
with ActivationAddition(["transformer.h.7.mlp"]).prepare(model) as hooked:
    steer = hooked(TensorDict({("positive", "input"): pos, ("negative", "input"): neg}, batch_size=1)).get(("steer", "transformer.h.7.mlp"))

# Cache activations
with ActivationCaching(r"transformer\.h\.\d+\.mlp", relative=False).prepare(model) as hooked:
    hooked(data)
    cache = hooked.hooking_context.cache

# Probing (needs ProbeManager, labels, step_type)
manager = ProbeManager(LogisticRegression, {}, compute_metrics)
with Probing("transformer.h.(0|5|10).mlp$", manager.probe_factory, additional_keys=["labels", "step_type"]).prepare(model, in_keys=["input_ids"], out_keys=["logits"]) as hooked:
    hooked(train_td)
    hooked(test_td)
```

### Weights

| Class | Use |
|-------|-----|
| `Pruning` | Zero params by `importance_callback`. Params: `amount_to_prune`, `skip_modules`, `modules_to_prune` |
| `Adapters` | Insert modules: `{path: (adapter, source, target)}` |
| `TaskVectors` | `get_task_vector`, `get_forget_vector`, `get_weights`, `compute_alpha` |

```python
from tdhook.weights import Pruning, Adapters, TaskVectors

with Pruning(importance_callback=fn, amount_to_prune=0.5).prepare(model) as hooked:
    hooked(inp)

with Adapters(adapters={"layer.5": (adapter, "layer.5", "layer.5")}).prepare(model) as hooked:
    hooked(data)
```
