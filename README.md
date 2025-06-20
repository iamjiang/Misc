## Installation

Install dependencies:
```bash
cd TR-Project
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

Select the kernel "TR-Project(Python 3.13.2) .venv/bin/python" in jupyter notebook
```

## Read Dataset

show the structure of data before reading the file
```bash
head -n 1 TRDataChallenge2023.txt | python3 -m json.tool
```

