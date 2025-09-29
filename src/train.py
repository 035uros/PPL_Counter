# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import MallDataset
from src.model import CrowdCounterCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(num_epochs=5, batch_size=4, learning_rate=1e-4):
    dataset = MallDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CrowdCounterCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, density_maps, num_people) in enumerate(dataloader):
            images, density_maps = images.to(device), density_maps.to(device)

            outputs = model(images)
            loss = criterion(outputs, density_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"[Epoch {epoch+1}] Batch {batch_idx}: images {images.shape}, density_maps {density_maps.shape}, num_people {num_people}"
                )

        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/crowd_counter.pth")
    print("Training complete. Model saved to checkpoints/crowd_counter.pth")


if __name__ == "__main__":
    train(num_epochs=2, batch_size=8, learning_rate=1e-4)
