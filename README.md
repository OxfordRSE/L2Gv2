
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

Clone the repository on your machine

```shell
git clone https://github.com/OxfordRSE/L2Gv2.git
```

Setup the virtual environment

1. Create and activate a virtual environment
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies

    ```shell
    pip install . --find-links https://data.pyg.org/whl/torch-{version}%2Bcpu.html
    ```

   For the above, select 2.5.1 for macOS and 2.4.1 for Linux. Note that this installs
   CPU versions of the dependencies. To install GPU versions, consult the
   [pytorch-geometric](https://pypi.org/project/torch-geometric/) documentation
   for the appropriate repository links, or visit https://data.pyg.org/whl to see
   all possible torch/GPU supported versions.

3. To build docs and for tests install the corresponding optional dependency sets

   ```shell
   pip install '.[tests]'
   pip install '.[docs]'
   ```

## License

## Contributors

The following people contributed to this project ([emoji key](https://allcontributors.org/docs/en/emoji-key)).


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
