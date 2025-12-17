# engine/model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_PLANES, POLICY_SIZE, BOARD_SIZE


class ChessNet(nn.Module):
    """
    Red neuronal para ajedrez con dos cabezas:
    - política: vector de tamaño POLICY_SIZE (4096)
    - valor: escalar entre -1 y 1
    """

    def __init__(self, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.policy_size = policy_size

        # Bloque convolucional básico
        self.conv1 = nn.Conv2d(NUM_PLANES, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Cabeza de política
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, self.policy_size),
        )

        # Cabeza de valor
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: tensor (batch, H, W, C) = (B, 8, 8, NUM_PLANES)
        Devuelve:
        - policy_logits: (batch, POLICY_SIZE)
        - value: (batch, 1) en rango [-1,1]
        """
        # Reordenar a formato (B, C, H, W) para Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Política
        policy_logits = self.policy_head(x)

        # Valor
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [-1, 1]

        return policy_logits, v

    def predict(self, board_tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        board_tensor: numpy o torch de forma (8,8,NUM_PLANES) o (B,8,8,NUM_PLANES).
        Devuelve policy_probs (softmax) y value.
        """
        if isinstance(board_tensor, torch.Tensor):
            x = board_tensor
        else:
            x = torch.from_numpy(board_tensor)

        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1,8,8,C)

        x = x.to(next(self.parameters()).device, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            logits, value = self.forward(x)
            # Softmax sobre dimensión de acciones
            policy = F.softmax(logits, dim=1)

        return policy, value
