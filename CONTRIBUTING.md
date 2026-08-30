# How to Contribute?

## Guidelines

The project dependencies are managed using `uv`, see the installation [guide](https://docs.astral.sh/uv/).
Use Python `>=3.11` (the same versions tested in CI are `3.11`, `3.12`, and `3.13`).

Additionally, install `just` to run the project shortcut commands.

## Dev Install

To install the dependencies:

```bash
uv sync --group dev --group conformance --extra hub
```

Before committing, install `pre-commit`:

```bash
uv run --group dev pre-commit install
```

To run the checks (`pre-commit` checks):

```bash
just checks
```

To run the tests (using `pytest`):

```bash
just tests
```

To build the wheel, install it into a fresh virtual environment, and run the
maintained seven-use-case workflow without importing the checkout:

```bash
just tests-wheel
```

Official-Lczero process conformance is deliberately opt-in. Pin the exact
binary, network, and version token reported by `lc0 --version`:

```bash
LC0_EXECUTABLE=/path/to/lc0 \
LC0_NETWORK=/path/to/network.pb.gz \
LC0_VERSION=v0.31.2 \
LC0_BACKEND=eigen \
just tests-live-lczero
```

The live check verifies both paths, matches the declared version against the
binary, records the network SHA-256 digest, and defaults to the portable
`eigen` CPU backend when `LC0_BACKEND` is omitted. It never upgrades public
root output to event or replayable evidence.

To build the documentation locally:

```bash
just docs
```

## Branches

Make a branch before opening a pull request to `main`.

## Scope gate

Core changes must either preserve lc0 model interoperability or add
chess-domain decision evidence. Hooks, attribution, probing, SAE/transcoder,
and coaching abstractions belong in downstream integrations unless a later
scope decision explicitly changes this boundary. Public API changes should be
demonstrated in a maintained notebook and documented through the generated API
reference.
