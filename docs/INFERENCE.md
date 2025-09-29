# 🔍 Inference Module (`src/inference.py`)

This module provides **inference functionality** for the crowd counting project.  
It is mainly used for testing the trained model on a **single batch** of images and visualizing predictions.

---

## 📂 File Overview

**File:** `src/inference.py`  
**Key function:** `test_single_batch(batch_size=4)`

---

## ⚙️ How It Works

1. **Device Configuration**  
   - Automatically checks if a **GPU (CUDA)** is available.
   - Falls back to **CPU** if CUDA is not found.

2. **Dataset Loading**  
   - Loads the **MallDataset** via `MallDataset()` from `src/data_loader.py`.
   - Uses a PyTorch `DataLoader` to fetch data in batches.

3. **Model Loading**  
   - Initializes the **CrowdCounterCNN** model (`src/model.py`).
   - Loads trained weights from:  
     ```
     checkpoints/crowd_counter.pth
     ```

4. **Batch Inference**  
   - Fetches the **first batch** of images, density maps, and ground-truth counts.
   - Passes the images through the model to obtain predicted density maps.
   - Counts are estimated by summing the density map pixels.

5. **Visualization**  
   - Displays **side-by-side comparison** of:
     - The **original input image** (with ground truth count).
     - The **predicted density map** (with predicted count).

---

## ▶️ Usage

You can call this function from anywhere (e.g., `main.py`) by importing it:

```python
from src.inference import test_single_batch

test_single_batch(batch_size=4)
````

Or run inference directly via `main.py`:

```bash
python main.py --mode inference
```

---

## 📊 Output Example

When running inference, you will see:

```
Device: cuda
Ground truth count: 29
Predicted count: 27.45
```

And a matplotlib visualization:

* **Left:** Original image with ground truth count.
* **Right:** Predicted density map with predicted count.

---

## ⚠️ Notes

* Ensure that a **trained model checkpoint** exists at `checkpoints/crowd_counter.pth`.
* Without this file, inference will fail.
* The default batch size is `4`, but you can adjust it when calling the function.

---

## 🔮 Future Extensions

* Allow random batch selection instead of always taking the first batch.
* Add an option to save visualizations instead of only displaying them.
* Support evaluation over the entire dataset (not just one batch).

