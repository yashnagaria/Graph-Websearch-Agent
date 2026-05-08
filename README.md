# Graph Websearch Agent

A custom web search and graph execution toolkit for flexible query workflows. This project includes search integration, model selection support, and local server scripts for Windows and Linux.

## Prerequisites

- Python 3.11 or later
- Conda or a compatible virtual environment tool
- Required Python packages from `requirements.txt`

## Environment Setup

1. Create a new virtual environment:
   ```bash
   conda create -n agent_env python=3.11 pip
   ```

2. Activate the environment:
   ```bash
   conda activate agent_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configure the Project

1. Open `config/config.yaml` in a text editor.
2. Provide the required search and model keys in the config file.
3. Save the updated configuration.

## Run the Project

- On Windows:
  ```powershell
  .\run_windows.ps1
  ```

- On Linux/macOS:
  ```bash
  chmod +x run_linux.sh
  ./run_linux.sh
  ```

## Run in Shell

To start the agent directly from the command line:
```bash
python -m app.app
```
Then enter your query when prompted.

## Notes

- This repository is intended for local experimentation and development.
- The nested project root contains the complete source tree and setup scripts.
