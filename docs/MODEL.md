# Model — Implementation & Reference

This document describes the CNN model implemented in `src/model.py`. It explains the design decisions, how the model operates, how to test it, and practical notes for training and extension. Use this file as the authoritative reference to include in the repository.

---

## Purpose

`model.py` provides a compact Convolutional Neural Network (CNN) that maps an RGB input image to a single-channel density map. The density map is intended to represent people density per pixel; summing all pixels yields an estimated person count for the image. The implementation is intentionally minimal and suitable as a starter architecture for prototyping people counting and density-map regression workflows.

---

## Design overview

* **Input:** single RGB image, shape `(B, 3, H, W)` where `B` is batch size. The starter test uses `H=480`, `W=640`.
* **Output:** single-channel density map, shape `(B, 1, H, W)`.
* **Model type:** fully convolutional network (no fully connected layers), which preserves spatial resolution.
* **Loss (training suggestion):** pixel-wise Mean Squared Error (MSE) between predicted density map and ground-truth density map; person count is obtained by summing the predicted density map.

The current architecture is deliberately small and easy to understand; it consists of a few convolutional layers with ReLU activations and a 1×1 final convolution to produce the density map.

---

## File contents — main components

### `CrowdCounterCNN` class

* Subclasses `torch.nn.Module`.
* Layers (example configuration in the repository):

  * `Conv2d(3, 16, kernel_size=9, padding=4)` → `ReLU`
  * `Conv2d(16, 32, kernel_size=7, padding=3)` → `ReLU`
  * `Conv2d(32, 16, kernel_size=7, padding=3)` → `ReLU`
  * `Conv2d(16, 8, kernel_size=7, padding=3)` → `ReLU`
  * `Conv2d(8, 1, kernel_size=1)` → final density map (no sigmoid; values are continuous)
* `forward(x)` applies layers sequentially and returns the final density map.

### Test routine (if run as script)

* The bottom of `model.py` contains a small test block that:

  * Builds the model
  * Runs a dummy input `torch.randn(1, 3, 480, 640)` through it
  * Prints input and output shapes for quick verification

---

## How to test locally

1. Activate the project virtual environment (see project README).
2. Run the model test:

```bash
python src/model.py
```

Expected output (example):

```
Input shape: torch.Size([1, 3, 480, 640])
Output shape: torch.Size([1, 1, 480, 640])
```

This confirms the forward pass works and the model preserves spatial dimensions.

---

## Quick usage snippets

### Instantiate and forward an image (PyTorch)

```python
import torch
from src.model import CrowdCounterCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdCounterCNN().to(device)
image = torch.randn(1, 3, 480, 640, device=device)  # example tensor
density = model(image)                              # (1, 1, 480, 640)
predicted_count = density.sum().item()
```

### Save / load model weights

```python
# save
torch.save(model.state_dict(), "crowdcounter.pth")

# load
model = CrowdCounterCNN()
model.load_state_dict(torch.load("crowdcounter.pth", map_location="cpu"))
model.eval()
```

---

## Training recommendations

* **Loss function:** `torch.nn.MSELoss()` between predicted density map and ground-truth density map.
* **Optimizer:** `torch.optim.Adam(model.parameters(), lr=1e-4)` as a good starting point.
* **Metrics:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on counts:

  * `MAE = mean(|pred_count - gt_count|)` over validation set.
  * `RMSE = sqrt(mean((pred_count - gt_count)^2))`.
* **Input preprocessing:** normalize images to float32 and scale to `[0, 1]` (or use ImageNet mean/std if fine-tuning pretrained backbones).
* **Batching:** use `torch.utils.data.DataLoader` with an appropriate batch size (CPU vs GPU).
* **Device:** training benefits greatly from a GPU; move model and tensors to `cuda` when available.

---

## Practical notes & limitations

* **Current model is minimal.** It works as a baseline and for debugging the training pipeline, but it is not state-of-the-art. For better performance consider:

  * Adding Batch Normalization (`nn.BatchNorm2d`) after conv layers.
  * Using pooling / strided convolutions and then learnable upsampling (deconvolution or bilinear upsample + conv) if memory/compute are constrained.
  * Leveraging pretrained backbones (e.g., ResNet features) as encoder and a small decoder to output a density map.
* **Output scale:** The network produces raw continuous values. There is no enforced normalization to ensure `sum(output) == gt_count`. Proper training with MSE on density maps usually learns correct scaling, but check predicted counts during validation and consider additional regularization if necessary.
* **Ground truth quality:** This repository started with a dataset that provides counts only (no x,y annotations). For per-pixel density supervision you need datasets with annotated head locations; then use a Gaussian kernel to produce ground-truth density maps.
* **Batch size & memory:** Dense per-pixel supervision has high memory cost for large images. You may need to crop/resize images or train on smaller patches.

---

## Suggested improvements (next steps)

* Replace this small hand-crafted CNN with a stronger encoder–decoder architecture (e.g., U-Net style or pretrained encoder + decoder).
* Add normalization layers and residual connections.
* Implement training loop (`train.py`) with learning rate scheduling, checkpoint saving, and per-epoch validation.
* Create a visualization utility that overlays predicted density map on original image for qualitative debugging.
* If you obtain point annotations, switch from uniform density maps to Gaussian-kernel density maps (see `generate_density_map` in `src/data_loader.py`).

---

## Dependencies

The model file relies on PyTorch. Make sure your environment meets the project requirements (see `requirements.txt`):

* `torch`
* `torchvision` (if using pretrained components later)

---

## Contact / provenance

This implementation is a compact, educational starter model intended for prototyping and pipeline verification. It is provided as part of the `PPL_Counter` repository and is meant to be extended for production use or research.