import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from TIISafetyNet.load_dataset import TIISafetyDataset
from TIISafetyNet.model import KeypointPredictionModel


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training Batches"):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation Batches"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    mse = total_loss / len(loader)
    return mse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset')
results_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results')

model = KeypointPredictionModel().to(device)

learning_rate = 5e-4
batch_size = 32
num_epochs = 20
window_size = 10

data_path = os.path.join(dataset_path, 'data.csv')

train_dataset = TIISafetyDataset(input_file=data_path, split="train", window_size=10)
val_dataset = TIISafetyDataset(input_file=data_path, split="val", window_size=10)
test_dataset = TIISafetyDataset(input_file=data_path, split="test", window_size=10)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_mse = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}")

test_mse = evaluate(model, test_loader, criterion)
print(f"Final Test MSE: {test_mse:.4f}")

save_path = os.path.join(results_path, "keypoint_model.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
