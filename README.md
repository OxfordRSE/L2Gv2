# l2gv2 - Local2Global

## Installation

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux  

To set up your environment, follow these steps

1. Create and activate a virtual environment
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
2. Install the dependencies

    ```shell
    pip install -r requirements.txt
The unified `requirements.in` file includes both shared and platform-specific dependencies with version constraints where necessary. To update dependencies, modify `requirements.in` and then recompile `requirements.txt`

```shell
pip-compile requirements.in --verbose
