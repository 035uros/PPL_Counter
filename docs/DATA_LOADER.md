# Data Loader - Overview

This file (`src/data_loader.py`) is responsible for **loading images and generating density maps** from the Mall dataset.  
It prepares the data so it can be fed into a CNN model for people counting.

---

## Technologies used

- **Python 3.13+** – main programming language
- **NumPy** – numerical operations and array handling
- **SciPy** (`scipy.io`) – reading `.mat` files
- **OpenCV (`cv2`)** – image loading and color conversion
- **Matplotlib** – visualizing images and density maps
- **glob** – file path management

---

## Main Components

### 1. Gaussian Kernel & Density Map Functions

```python
def gaussian_kernel(size=15, sigma=4)
````

* Generates a **2D Gaussian kernel** for creating density maps.

```python
def generate_density_map(image_shape, points, kernel_size=15, sigma=4)
```

* Converts **x,y coordinates of people** into a density map.
* The sum of all pixels equals the number of people.
* Currently not used with the Mall dataset because exact coordinates are missing.

---

### 2. `MallDataset` Class

**Purpose:** Handles loading images and generating **dummy density maps** based on the number of people (`counts`) from `mall_gt.mat`.

#### `__init__(self, dataset_path="datasets/mall_dataset/")`

* Loads ground truth data (`frame` and `count`) from `.mat` file.
* Collects all image file paths in `frames/`.

#### `__len__(self)`

* Returns the total number of images in the dataset.

#### `__getitem__(self, idx)`

* Loads the image at index `idx`.
* Reads the number of people (`num_people`) for that image.
* Generates a **uniform density map** (sum = number of people) for testing purposes.
* Returns the **image** and **density map**.

---

### 3. Test / Visualization Section

```python
if __name__ == "__main__":
```

* Tests the data loader by loading the **first image** and its density map.
* Prints:

  * Image shape
  * Density map sum (should match number of people)
* Visualizes:

  * Original image
  * Density map as a heatmap

This allows quick verification that the loader works correctly.

---

### Notes

* Current density maps are **uniform** because the Mall dataset `.mat` file does not contain exact x,y coordinates.
* This setup is sufficient for **testing model training pipelines**.
* Once exact coordinates are available, `generate_density_map` can be used to create **Gaussian density maps** for better accuracy.

---

### Usage

```python
from src.data_loader import MallDataset

dataset = MallDataset()
img, density_map = dataset[0]
```

This prepares data ready for **model training or evaluation**.
