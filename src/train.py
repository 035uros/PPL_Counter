# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MallDataset
from model import CrowdCounterCNN

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_epochs=5, batch_size=4, learning_rate=1e-4):
    # Load dataset
    dataset = MallDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = CrowdCounterCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, density_maps, num_people) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)            # [B, 3, H, W]
            density_maps = density_maps.to(device)  # [B, 1, H, W]
            num_people = num_people.to(device)    # [B]

            # Debug: shapes
            if batch_idx % 10 == 0:  # every 10 batches
                print(f"[Epoch {epoch+1}] Batch {batch_idx}: images {images.shape}, density_maps {density_maps.shape}, num_people {num_people}")

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, density_maps)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/crowd_counter.pth")
    print("Training complete. Model saved to checkpoints/crowd_counter.pth")


if __name__ == "__main__":
    train(num_epochs=2, batch_size=2, learning_rate=1e-4)
