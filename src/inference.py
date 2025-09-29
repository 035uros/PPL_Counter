# src/inference.py
import torch
from src.data_loader import MallDataset
from src.model import CrowdCounterCNN  # <-- obavezno ime klase iz model.py
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_single_batch(batch_size=4):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = MallDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = CrowdCounterCNN().to(device)
    model.load_state_dict(
        torch.load("checkpoints/crowd_counter.pth", map_location=device)
    )
    model.eval()

    # Take first batch
    images, density_maps, num_people = next(iter(dataloader))
    images = images.to(device)
    density_maps = density_maps.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Compute counts
    predicted_counts = outputs.view(outputs.size(0), -1).sum(dim=1).cpu().numpy()
    gt_counts = num_people.numpy()

    for i in range(len(predicted_counts)):
        print(f"Ground truth count: {gt_counts[i]}")
        print(f"Predicted count: {predicted_counts[i]:.2f}")

        # visualize
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title(f"Original Image (GT: {gt_counts[i]})")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(outputs[i][0].cpu().numpy(), cmap="jet")
        plt.title(f"Density Map (Pred: {predicted_counts[i]:.2f})")
        plt.axis("off")
        plt.show()
