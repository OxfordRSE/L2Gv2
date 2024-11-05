
[![Pylint](https://github.com/lotzma/L2Gv2/actions/workflows/pylint.yml/badge.svg)](https://github.com/lotzma/L2Gv2/actions/workflows/pylint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)

# l2gv2 - Local2Global

## Overview

## Documentation

Full documentation available [here](https://l2gv2.readthedocs.io/en/latest/)


## Setup


**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux  

Clone the repository on your machine

```shell
git clone https://github.com/lotzma/L2Gv2.git
```

Setup the virtual environment

1. Create and activate a virtual environment
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies

    ```shell
    pip install -r requirements.txt
    ```

The unified `requirements.in` file includes both shared and platform-specific dependencies with version constraints where necessary. To update dependencies, modify `requirements.in` and then recompile `requirements.txt`

```shell
pip-compile requirements.in --verbose
```

## License

## Contributors

The following people contributed to this project ([emoji key](https://allcontributors.org/docs/en/emoji-key)). 

<!-- ALL-CONTRIBUTORS-LIST:START -->
<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/lotzma">
            <img src="https://avatars.githubusercontent.com/u/22026207?v=4?v=4?s=100" width="100px;" alt="Martin Lotz"/>
            <br /><sub><b>Martin Lotz</b></sub>
            <br />🤔 📆 💻 💡
        </a>
    </td>  
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.