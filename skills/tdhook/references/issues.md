# GitHub Issues & Solutions

Common issues from the tdhook repository and how to resolve them.

## Where to Look

- **Issue tracker**: [github.com/Xmaster6y/tdhook/issues](https://github.com/Xmaster6y/tdhook/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Xmaster6y/tdhook/discussions) (if enabled)

## Common Patterns

| Issue | Solution |
|-------|----------|
| `KeyError` on TensorDict | Use tuple keys for nested: `("attr", "input")`, `("baseline", "input")` |
| HuggingFace `in_keys`/`out_keys` | Pass `in_keys=["input_ids"]`, `out_keys=["logits"]` to `prepare()` |
| BilinearProbeManager h1≠h2 | Call `manager.before_all()` before forwards, `manager.after_all()` after |
| Module path not found | Use relative paths or `relative=False`; check `resolve_submodule_path` patterns in api.md |
| Probing `step_type` missing | Include `additional_keys=["labels", "step_type"]` and pass both in TensorDict |

## Contributing

When filing an issue, include: tdhook version, model type (nn.Module vs TensorDictModule), minimal repro, and full traceback.
