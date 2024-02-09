# How to Contribute?

## Guidelines

The project dependencies are managed using `poetry`, see their installation [guide](https://python-poetry.org/docs/). For even more stability, I recommend using `pyenv` or python `3.9.16`.

Additionally, to make your life easier, install `make` to use the shortcut commands.

## Dev Install

To install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
poetry install --with dev
```

Before committing, install `pre-commit`:

```bash
poetry run pre-commit install
```

To run the checks (`pre-commit` checks):

```bash
make checks
```

To run the tests (using `pytest`):

```bash
make tests
```

## Branches

Make a branch before making a pull request to `develop`.
