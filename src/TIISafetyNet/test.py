import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from TIISafetyNet.load_dataset import TIISafetyDataset
from TIISafetyNet.model import KeypointPredictionModel


def test(input_sequence):
    """
    Funzione per testare manualmente il modello con una sequenza di input.
    :param input_sequence: np.array o lista con la sequenza di input (window_size - 1, num_features)
    :return: None, stampa la predizione
    """
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    print("Predicted next record:", prediction.cpu().numpy())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results')
save_path = os.path.join(results_path, "keypoint_model.pth")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset', 'data.csv')

model = KeypointPredictionModel().to(device)
model.eval()
model.load_state_dict(torch.load(save_path))

batch_size = 32

train_dataset = TIISafetyDataset(input_file=data_path, split="train", window_size=10)
val_dataset = TIISafetyDataset(input_file=data_path, split="val", window_size=10)
test_dataset = TIISafetyDataset(input_file=data_path, split="test", window_size=10)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

manual_test(input_sequence)
