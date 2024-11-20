# Investigation of Redox Ratio of 2 Photon Imaging of Intestinal Organoids

## Prerequesites
- python >= 3.10
- pip or uv (https://uv.readthedocs.io/en/latest/)
- git

## Getting Started
1. download git repository onto your local machine

```bash
git clone https://github.com/phamousq/2photonOrganoid.git
cd 2photonOrganoid
```

2. install dependencies
```bash
pip install .
```

3. Set up your virtual environment
```bash
uv venv && source .venv/bin/activate && uv sync
```

- on windows with pip
```bash
python -m venv venv
.\venv\Scripts\activate
```

- on macos/linus with pip
```bash
python3 -m venv venv
source venv/bin/activate
```

4. run the program
```bash
python main.py
```