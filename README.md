# Environment Set up (pip or conda)
Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

## Option 1: use pip (recommended)
- Use the supplied file `requirements.txt` to create a new environment with pip:
  
```sh
python -m venv venv
source venv/bin/activate # on Linux/macOS
venv\Scripts\activate # on Windows
pip install -r requirements.txt


