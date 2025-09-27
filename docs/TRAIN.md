
# `train.py` - Overview

**Description:**
This script trains a CNN model for crowd counting on images using **density maps**. The model learns to predict the density of people in an image, using ground truth coordinates and counts from a `.mat` file.

---

## Input Data

* Dataset: `datasets/mall_dataset/`

  * `frames/` — folder containing all images (`.jpg`)
  * `mall_gt.mat` — ground truth file with the following keys:

    * `frame` — array containing people coordinates for each image
    * `count` — number of people in each image

* Parameters you can set when calling the `train()` function:

  * `num_epochs` — number of training epochs (default: 5)
  * `batch_size` — batch size for training (default: 4)
  * `learning_rate` — learning rate for the Adam optimizer (default: 1e-4)

---

## Output

* The trained model is saved in the `checkpoints/` folder as:

  ```text
  checkpoints/crowd_counter.pth
  ```

* During training, the console displays:

  * Epoch number
  * Batch index
  * Shape of input images and density maps
  * Number of people in the current batch
  * Average MSE loss per epoch

---

## Usage

```bash
python src/train.py
```

* The script automatically loads the dataset, creates the model, trains it, and saves the weights.
* Recommendation: Use smaller batch sizes or resize images when training on CPU to improve speed.

---

## Notes / Optimization Tips

* Density maps are currently generated **on the fly** from coordinates, which can slow down training on larger datasets.
* To speed up training:

  * Reduce batch size or image dimensions
  * Pre-generate density maps and load them directly from files
  * Use GPU if available
