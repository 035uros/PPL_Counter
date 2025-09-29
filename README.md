# PPL_Counter

Project for people counting and heat/density map generation.

## Project Structure

```
PPL_Counter/
│   .gitignore
│   requirements.txt
│   main.py
│
├── checkpoints/
│   └── crowd_counter.pth
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
- Train the model using `src/train.py`. Progress described [here](docs/TRAIN.md).

## How to work with the project


The project supports two modes: **training** and **inference**, and allows additional configuration for evaluation.

---

## ▶️ Usage

Run the project with:

```bash
python main.py --mode [train|inference] [options]
```

---

### 1. Training

To start training the model:

```bash
python main.py --mode train
```

This will:

* Train the model on the Mall dataset.
* Save checkpoints in the `checkpoints/` folder.

---

### 2. Inference

You can test the trained model on the dataset with:

```bash
python main.py --mode inference
```

#### Additional Options for Inference:

* `--num_samples N`
  Run inference only on **N random samples** instead of the entire dataset.
  Example:

  ```bash
  python main.py --mode inference --num_samples 10
  ```

* `--visualize 1`
  Display the image with its predicted density map overlay.
  Example:

  ```bash
  python main.py --mode inference --visualize 1
  ```

You can combine both:

```bash
python main.py --mode inference --num_samples 5 --visualize 1
```

---

## 📊 Evaluation Metrics

After inference, the following metrics are calculated:

* **MAE (Mean Absolute Error):**
  Measures how far the predictions are from ground truth on average.

  * Lower MAE = better performance.
  * Example: `MAE: 9.74` means the model is off by about 10 people on average.

* **MSE (Mean Squared Error):**
  Penalizes larger errors more heavily.

  * Lower MSE = better performance.
  * Example: `MSE: 134.72` means there are occasional larger miscounts.

---

## 📸 Example Output

When running inference:

```
Ground truth count: 29.0
Predicted count: 27.4

Ground truth count: 35.0
Predicted count: 31.7

📊 Evaluation results:
MAE: 9.74
MSE: 134.72
```

With visualization enabled, the program will display the original frame and its density map side by side.

---

## 🔮 Future Improvements

* Add support for more datasets.
* Implement better CNN architectures (e.g., CSRNet, MCNN).
* Deploy the model as a web service for real-time crowd counting.

---



## OBJECTIONS

Where to train? Explore Google Cloud, Microsoft Azure ML Studio, Amazon Sagemaker

## ✅ Notes

- Make sure you download and place the **Mall dataset** in `datasets/mall_dataset/`.
- A pretrained model should be available at `checkpoints/crowd_counter.pth`.
- For best results, train the model for more epochs (`--mode train`).
- Python 3.13+ is recommended
- Tested with PyTorch 2.8+ (CPU version)
- Make sure the virtual environment is activated before running scripts