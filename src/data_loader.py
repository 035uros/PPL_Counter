# src/data_loader.py

import os
import scipy.io as sio
import numpy as np
import cv2
from glob import glob
import torch


# ----------------------------
# Helper function: generate Gaussian density map
# ----------------------------
def gaussian_kernel(size=15, sigma=4):
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def generate_density_map(image_shape, points, kernel_size=15, sigma=4):
    h, w = image_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    kernel = gaussian_kernel(kernel_size, sigma)
    k = kernel_size // 2

    for point in points:
        x, y = min(w - 1, max(0, int(point[0]))), min(h - 1, max(0, int(point[1])))
        x1, y1 = max(0, x - k), max(0, y - k)
        x2, y2 = min(w, x + k + 1), min(h, y + k + 1)

        kx1, ky1 = k - (x - x1), k - (y - y1)
        kx2, ky2 = k + (x2 - x), k + (y2 - y)

        density_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    return density_map


# ----------------------------
# Main DataLoader
# ----------------------------
class MallDataset:
    def __init__(self, dataset_path="datasets/mall_dataset/"):
        self.dataset_path = dataset_path
        self.frames_path = os.path.join(dataset_path, "frames")
        self.gt_file = os.path.join(dataset_path, "mall_gt.mat")

        # Load ground truth
        gt_data = sio.loadmat(self.gt_file)
        self.frames_info = gt_data["frame"].flatten()

        # Directly use count from mat file
        self.counts = gt_data["count"].flatten()
        self.counts = self.counts.astype(np.int32)

        # Get all image files
        self.image_files = sorted(glob(os.path.join(self.frames_path, "*.jpg")))

        print("Keys:", gt_data.keys())
        print("Frame shape:", gt_data["frame"].shape)
        print("Counts sample:", self.counts[:10])
        print("Number of images:", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # extract points for density map
        frame_data = self.frames_info[idx]
        points = frame_data["loc"][0]
        points_list = [np.array(p).flatten().tolist() for p in points]

        h, w = img.shape[:2]
        density_map = generate_density_map((h, w), points_list)

        # convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        density_tensor = torch.from_numpy(density_map).unsqueeze(0).float()
        num_people = torch.tensor(self.counts[idx], dtype=torch.float)

        return img_tensor, density_tensor, num_people


# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    dataset = MallDataset()
    img, density, count = dataset[0]

    print("Image shape:", img.shape)
    print("Density map sum (count estimate):", density.sum())
    print("Ground truth count:", count)
