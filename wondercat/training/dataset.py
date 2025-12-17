# training/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    """
    Dataset que carga uno o varios .npz de self-play.

    Cada fichero debe contener:
      - states: (N, 8, 8, C)
      - policies: (N, POLICY_SIZE)
      - values: (N, 1)
    """

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.npz"))
        if not files:
            raise ValueError(f"No se encontraron ficheros .npz en {data_dir}")

        states_list: List[np.ndarray] = []
        policies_list: List[np.ndarray] = []
        values_list: List[np.ndarray] = []

        for f in files:
            arr = np.load(f)
            states_list.append(arr["states"])
            policies_list.append(arr["policies"])
            values_list.append(arr["values"])

        states = np.concatenate(states_list, axis=0)
        policies = np.concatenate(policies_list, axis=0)
        values = np.concatenate(values_list, axis=0)

        self.states = torch.from_numpy(states).float()      # (N, 8, 8, C)
        self.policies = torch.from_numpy(policies).float()  # (N, A)
        self.values = torch.from_numpy(values).float()      # (N, 1)

        if not (len(self.states) == len(self.policies) == len(self.values)):
            raise ValueError("Dimensiones inconsistentes en el dataset.")

        print(
            f"[ChessDataset] Cargadas {len(self.states)} posiciones "
            f"desde {len(files)} ficheros."
        )

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.policies[idx], self.values[idx]


def load_data(
    data_dir: str | Path,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Crea un DataLoader a partir de los ficheros .npz de self-play.
    """
    dataset = ChessDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
