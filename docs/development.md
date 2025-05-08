# Development

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux

**Clone the repository** on your machine

```shell
git clone https://github.com/OxfordRSE/L2Gv2.git
```

## Package management using `uv`

We use [`uv`](https://docs.astral.sh/uv/) for Python package management. You
can install it on macOS or Linux using `brew install uv`. Alternatively, you
can use uv's [installation script](https://docs.astral.sh/uv/#installation).

To run arbitrary commands in the `uv` generated Python environment, first
install all dependencies:
```shell
uv sync --all-extras --dev
```
(This is done automatically if you are using the nox targets)

## Running tasks with `nox`

[nox](https://nox.thea.codes) simplifies Python testing, particularly across
multiple Python versions. We provide a [noxfile.py](noxfile.py), which allows you
to run tests and perform linting with one command. You'll first need
to install nox:

```shell
brew install nox      # macOS
pipx install nox      # with pipx
sudo apt install nox  # debian
sudo dnf install nox  # fedora
uv tool install nox   # with uv
```

To run the tests and linting with
[pylint](https://pylint.readthedocs.io/en/stable/) and
[ruff](https://docs.astral.sh/ruff/):

```shell
nox
```

To display a list of tasks:

```shell
nox --list
```

To run only a task, such as `lint`, run `nox -s lint`.

## Using pre-commit

For development, please install the pre-commit hook that helps lint and
autoformat on every commit:

```shell
brew install pre-commit     # macOS
pipx install pre-commit     # with pipx
sudo apt install pre-commit # debian
sudo dnf install pre-commit # fedora
uv tool install pre-commit  # with uv
```

To setup pre-commit hooks, run `pre-commit install` once in the repository; this
will ensure that checks run before every commit. The pre-commit hooks are setup
to do basic style and formatting checks, and lint checks using `ruff` and
`pylint`. The configuration for these tools are in the `pyproject.toml` file.
These checks also run in GitHub Actions to ensure code quality.

## Documentation

The project is setup to generate documentation with
[Jupyter-book](https://jupyterbook.org).

Documentation is automatically generated on pushes to the `main` branch at:
https://l2gv2.readthedocs.io.

The package documentation can be generated locally:
```shell
nox -s docs
```

Then open `docs/_build/html/index.html` in your browser:

```shell
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
```