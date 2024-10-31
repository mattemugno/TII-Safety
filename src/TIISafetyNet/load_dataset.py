from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_sequences_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def timestamp_to_numeric(base_date_str):
    return int(datetime.strptime(base_date_str, '%Y-%m-%d_%H%M%S').timestamp())


def offset_to_numeric(offset):
    return float(offset.split('_')[0])


def split_sequences(df, window_size):
    input_target_pairs = []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size]

        input_sequence = window.iloc[:-1].to_numpy()
        target_sequence = window.iloc[-1].to_numpy()

        input_target_pairs.append((input_sequence, target_sequence))
    return input_target_pairs


class TIISafetyDataset(Dataset):
    def __init__(self, input_file, split, window_size=10):
        self.split = split
        self.window_size = window_size
        self.input_target_pairs = []

        df = load_sequences_from_csv(input_file)

        feature_columns = df.columns[2:]
        df = df[feature_columns]

        self.input_target_pairs = split_sequences(df, self.window_size)

        train_data, temp_data = train_test_split(self.input_target_pairs, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        if self.split == "train":
            self.input_target_pairs = train_data
        elif self.split == "val":
            self.input_target_pairs = val_data
        elif self.split == "test":
            self.input_target_pairs = test_data

    def __len__(self):
        return len(self.input_target_pairs)

    def __getitem__(self, index):
        input_sequence, target_sequence = self.input_target_pairs[index]

        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)

        return {
            "input": input_tensor,
            "target": target_tensor
        }
