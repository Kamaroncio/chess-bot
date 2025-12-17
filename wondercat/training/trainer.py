# training/trainer.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine.model import ChessNet
from .dataset import load_data


@dataclass
class TrainingConfig:
    """
    Configuración de entrenamiento.
    """
    data_dir: str | Path
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    l2_reg: float = 1e-4
    num_workers: int = 0
    device: str | None = None  # 'cuda', 'cpu', o None para auto
    num_threads: int = 2       # <<< NUEVO: nº máximo de hilos de CPU


def _limit_threads(num_threads: int) -> None:
    """
    Limita el número de hilos usados por BLAS/PyTorch para no saturar la CPU.
    """
    num_threads = max(1, int(num_threads))

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    # PyTorch
    torch.set_num_threads(num_threads)
    # Hilos para tareas internas (inter-op); la mitad suele ser razonable
    torch.set_num_interop_threads(max(1, num_threads // 2))


def train_model(
    model: ChessNet,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """
    Entrena el modelo ChessNet con datos de self-play.

    Devuelve un diccionario con estadísticas simples (por ahora).
    """
    if config.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # Limitar hilos ANTES de empezar a entrenar
    _limit_threads(config.num_threads)

    model.to(device)

    loader: DataLoader = load_data(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,  # 0 = no procesos extra
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    stats = {
        "epochs": config.epochs,
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
    }

    model.train()

    for epoch in range(config.epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for states, target_policies, target_values in loader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            logits, values = model(states)  # logits: (B, A), values: (B,1)

            # Política: cross-entropy con distribución objetivo π
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

            # Valor: MSE entre z y v
            values = values.squeeze(-1)
            target_values = target_values.squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(values, target_values)

            # Regularización L2 manual (además del weight_decay si quieres)
            l2 = 0.0
            if config.l2_reg > 0:
                for p in model.parameters():
                    l2 += torch.sum(p * p)
                l2 = config.l2_reg * l2

            loss = (
                config.policy_loss_weight * policy_loss
                + config.value_loss_weight * value_loss
                + l2
            )

            loss.backward()
            optimizer.step()

            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            epoch_total_loss += float(loss.item())
            num_batches += 1

        epoch_policy_loss /= max(num_batches, 1)
        epoch_value_loss /= max(num_batches, 1)
        epoch_total_loss /= max(num_batches, 1)

        stats["policy_loss"].append(epoch_policy_loss)
        stats["value_loss"].append(epoch_value_loss)
        stats["total_loss"].append(epoch_total_loss)

        print(
            f"[Train] Epoch {epoch+1}/{config.epochs} "
            f"policy_loss={epoch_policy_loss:.4f} "
            f"value_loss={epoch_value_loss:.4f} "
            f"total_loss={epoch_total_loss:.4f}"
        )

    return stats
