# training/self_play.py

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import chess

from engine import ChessGame
from engine.game import index_to_move
from engine.mcts import MCTS
from engine.model import ChessNet
from engine.config import POLICY_SIZE


@dataclass
class SelfPlayConfig:
    """
    Configuración básica para generar datos de self-play.
    """
    num_games: int = 10
    num_simulations: int = 200
    temperature: float = 1.0
    temp_drop_move: int = 15  # después de este movimiento, se juega casi determinista
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    seed: int | None = None


def _select_action(pi: np.ndarray, temperature: float) -> int:
    """
    Selecciona una acción a partir de la distribución pi y la temperatura.
    - temperature -> 0: modo casi determinista (argmax).
    - temperature = 1: muestreo normal.
    """
    if temperature <= 1e-6:
        return int(np.argmax(pi))

    # Evitar problemas de numérica
    pi = np.asarray(pi, dtype=np.float64)
    if pi.sum() <= 0:
        return int(np.argmax(pi))

    # Ajuste por temperatura
    pi = pi ** (1.0 / temperature)
    s = pi.sum()
    if s <= 0:
        return int(np.argmax(pi))
    pi /= s

    return int(np.random.choice(len(pi), p=pi))


def _game_to_targets(
    states: List[np.ndarray],
    policies: List[np.ndarray],
    players: List[bool],
    final_board: chess.Board,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte la secuencia de estados de una partida en tensores de entrenamiento:
    X: estados (T, 8, 8, C)
    P: políticas (T, POLICY_SIZE)
    Z: valores (T, 1) en [-1, 1]
    """
    outcome = final_board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        # tablas o sin resultado claro: valor 0 para todos
        z_val = 0.0
    else:
        # winner: True blancas, False negras
        winner = outcome.winner
        # para cada jugada, +1 si gana ese color, -1 si pierde
        z_vals = []
        for pl in players:
            z_vals.append(1.0 if pl == winner else -1.0)
        values = np.array(z_vals, dtype=np.float32).reshape(-1, 1)

        X = np.stack(states, axis=0).astype(np.float32)
        P = np.stack(policies, axis=0).astype(np.float32)
        return X, P, values

    # caso tablas (o sin outcome):
    X = np.stack(states, axis=0).astype(np.float32)
    P = np.stack(policies, axis=0).astype(np.float32)
    Z = np.zeros((len(states), 1), dtype=np.float32)
    return X, P, Z


def generate_self_play_data(
    model: ChessNet,
    output_dir: str | Path,
    config: SelfPlayConfig,
    device: torch.device | None = None,
) -> None:
    """
    Genera partidas de self-play usando MCTS + red y guarda los datos de entrenamiento
    en ficheros .npz en output_dir.

    Cada partida produce un archivo: selfplay_timestamp_gameIdx.npz
    con:
      - states: (T, 8, 8, C)
      - policies: (T, POLICY_SIZE)
      - values: (T, 1)
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for game_idx in range(config.num_games):
        game = ChessGame()  # tablero inicial
        board = game.board

        mcts = MCTS(
            model=model,
            num_simulations=config.num_simulations,
            cpuct=1.5,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            device=device,
        )

        states: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        players: List[bool] = []

        move_number = 0

        while not game.is_game_over():
            # estado actual
            state_tensor = game.to_tensor()
            states.append(state_tensor)
            players.append(game.turn)

            # ejecutamos MCTS desde este estado
            pi, root = mcts.run(board, add_dirichlet_noise=True)

            # temperatura: alta al principio, baja luego
            if move_number >= config.temp_drop_move:
                temp = 1e-3
            else:
                temp = config.temperature

            action_idx = _select_action(pi, temp)
            policies.append(pi.copy())

            move = index_to_move(action_idx, board)
            if move is None or move not in board.legal_moves:
                # En caso de rareza numérica, elegimos la jugada más visitada legal
                legal_actions = list(root.children.keys())
                if not legal_actions:
                    break
                best_action = max(
                    legal_actions,
                    key=lambda a_idx: root.children[a_idx].visit_count,
                )
                move = index_to_move(best_action, board)
                if move is None or move not in board.legal_moves:
                    # fallback extremo: jugada legal cualquiera
                    move = next(iter(board.legal_moves))

            game.apply_move(move)
            board = game.board
            move_number += 1

        # partida terminada, construimos targets
        X, P, Z = _game_to_targets(states, policies, players, board)

        timestamp = int(time.time())
        filename = output_dir / f"selfplay_{timestamp}_{game_idx}.npz"
        np.savez_compressed(
            filename,
            states=X,
            policies=P,
            values=Z,
        )
        print(f"[Self-Play] Partida {game_idx+1}/{config.num_games} guardada en {filename}")
