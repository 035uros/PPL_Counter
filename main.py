import torch
import numpy as np
import argparse
from src.train import train
from src.inference import test_single_batch
from src.data_loader import MallDataset
import matplotlib.pyplot as plt

print("✅ PyTorch check:", torch.__version__)
print("✅ NumPy check:", np.__version__)

def main():
    parser = argparse.ArgumentParser(description="Crowd Counting Project")
    parser.add_argument("--mode", choices=["train", "inference"], required=True,
                        help="Choose mode: train or inference")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to test (default: all dataset)")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Whether to show images and density maps (1=yes, 0=no)")
    args = parser.parse_args()

    if args.mode == "train":
        train(num_epochs=2, batch_size=8, learning_rate=1e-4)

    elif args.mode == "inference":
        # Load dataset
        dataset = MallDataset()

        # Load model
        from src.model import CrowdCounterCNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CrowdCounterCNN().to(device)
        model.load_state_dict(torch.load("checkpoints/crowd_counter.pth", map_location=device))
        model.eval()

        gt_counts = []
        pred_counts = []

        # Decide how many samples to process
        num_samples = args.num_samples if args.num_samples else len(dataset)
        print(f"🔎 Running inference on {num_samples} samples")

        for i in range(num_samples):
            img, density_map, gt_count = dataset[i]   # Get image, density map, ground truth count
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                pred_count = output.sum().item()

            print(f"[{i}] Ground truth count: {gt_count:.1f}")
            print(f"[{i}] Predicted count: {pred_count:.2f}")

            gt_counts.append(gt_count)
            pred_counts.append(pred_count)

            # Optional visualization
            if args.visualize == 1:
                img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                density_np = density_map.squeeze(0).cpu().numpy()
                pred_density_np = output.squeeze(0).squeeze(0).cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(density_np, cmap="jet")
                plt.title(f"GT Density (count={gt_count:.1f})")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_density_np, cmap="jet")
                plt.title(f"Predicted Density (count={pred_count:.2f})")
                plt.axis("off")

                plt.show()

        # Compute MAE and MSE
        gt_counts = np.array(gt_counts)
        pred_counts = np.array(pred_counts)

        mae = np.mean(np.abs(gt_counts - pred_counts))
        mse = np.mean((gt_counts - pred_counts) ** 2)

        print(f"\n📊 Evaluation results:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")


if __name__ == "__main__":
    main()
