# How to Contribute?

## Guidelines

The project dependencies are managed using `uv`, see the installation [guide](https://docs.astral.sh/uv/).
Use Python `>=3.10` (the same versions tested in CI are `3.10`, `3.11`, and `3.12`).

Additionally, install `just` to run the project shortcut commands.

## Dev Install

To install the dependencies:

```bash
uv sync
```

Before committing, install `pre-commit`:

```bash
uv run pre-commit install
```

To run the checks (`pre-commit` checks):

```bash
just checks
```

To run the tests (using `pytest`):

```bash
just tests
```

To build the documentation locally:

```bash
just docs
```

## Branches

Make a branch before opening a pull request to `main`.
