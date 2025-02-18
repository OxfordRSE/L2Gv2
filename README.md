
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Pylint](https://github.com/lotzma/L2Gv2/actions/workflows/pylint.yml/badge.svg)](https://github.com/lotzma/L2Gv2/actions/workflows/pylint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# l2gv2 - Local2Global

## Overview

## Documentation

Full documentation available [here](https://l2gv2.readthedocs.io/en/latest/)

## Setup

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux

**Clone the repository** on your machine

```shell
git clone https://github.com/OxfordRSE/L2Gv2.git
```

Create and activate a **virtual environment**

```shell
python3 -m venv .venv
source .venv/bin/activate
```

**Install the dependencies**

```shell
pip install '.[dev,docs]'
```

This will install dependencies, including [pytorch](https://pytorch.org) and
[pytorch-geometric](https://pyg.org). For macOS, CPU version of pytorch will
be installed, whereas for Linux, a GPU version targeting the latest CUDA
release will be installed. Installation of alternate or older CUDA versions
may be supported in the future.

To simplify testing for developers, we provide a [Makefile](Makefile), which
allows you to run the above steps and test with one command:

```shell
make
```

There are additional Makefile targets to automate tasks:

- `make test`: runs testing framework
- `make lint`: runs pylint
- `make format`: autoformats code using ruff
- `make ruff-checks`: lints and checks that code is formatted using ruff

Note that the above commands will install development and documentation
dependencies. If you are only using this library as a dependency, use:

```shell
pip install git+https://github.com/OxfordRSE/L2Gv2
```

## License

This project is licensed under the [MIT](LICENSE) license.

## Contributors

The following people contributed to this project ([emoji key](https://allcontributors.org/docs/en/emoji-key)).


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
