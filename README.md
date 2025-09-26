# PPL_Counter

Project for people counting and heat/density map generation.

## Project Structure

```
PPL_Counter/
│   .gitignore
│   requirements.txt
│   main.py
│
├── datasets/
│   └── mall_dataset/
│
│── docs/
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── model.py
    └── train.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/035uros/PPL_Counter.git
cd PPL_Counter
```

### 2. Set up Python virtual environment

**Windows PowerShell:**
```powershell
# Create virtual environment
py -m venv venv

# Allow script execution temporarily
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\venv\Scripts\activate
```

**Command Prompt (CMD):**
```cmd
py -m venv venv
venv\Scripts\activate.bat
```

After activation, you should see `(venv)` in front of your prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python main.py
```

You should see something like:
```
✅ PyTorch check: 2.x.x+cpu
✅ NumPy check: 1.x.x
```

### 5. Dataset

Place the Mall dataset inside `datasets/mall_dataset/`.
Dataset origin can be found [here](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html). Additional info [here](Datasets/mall_dataset/MALL_DATASET.md).

The folder should include:
- `frames/`
- `mall_gt.mat`
- `mall_feat.mat`
- `perspective_roi.mat`

## Next Steps

- Implement data loading and preprocessing in `src/data_loader.py`. Progress described [here](docs/DATA_LOADER.md).
- Implement CNN model for density map regression in `src/model.py`. Progress described [here](docs/MODEL.md).
- Train the model using `src/train.py`

## Notes

- Python 3.13+ is recommended
- Tested with PyTorch 2.8+ (CPU version)
- Make sure the virtual environment is activated before running scripts