# run_selfplay.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from engine.model import ChessNet
from training.self_play import SelfPlayConfig, generate_self_play_data


def _limit_threads(num_threads: int) -> None:
    """
    Limita hilos para que el self-play no se coma toda la CPU.
    """
    import torch as _torch

    num_threads = max(1, int(num_threads))

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    _torch.set_num_threads(num_threads)
    _torch.set_num_interop_threads(max(1, num_threads // 2))


def main():
    parser = argparse.ArgumentParser(description="Generar datos de self-play para ajedrez.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay",
        help="Directorio donde guardar los ficheros .npz de self-play.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al modelo .pt a usar para self-play (opcional, si no se usa modelo sin entrenar).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Número de partidas de self-play a generar.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=64,   # <<< MENOS SIMULACIONES POR JUGADA = MENOS CPU
        help="Número de simulaciones de MCTS por jugada.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura inicial para selección de jugadas.",
    )
    parser.add_argument(
        "--temp-drop-move",
        type=int,
        default=15,
        help="Después de este movimiento, la IA juega casi determinista.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,    # <<< Nº máximo de hilos durante self-play
        help="Número máximo de hilos de CPU que puede usar PyTorch y BLAS.",
    )
    args = parser.parse_args()

    # Limitar hilos antes de crear el modelo y lanzar MCTS
    _limit_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet()
    model.to(device)

    if args.model is not None:
        model_path = Path(args.model)
        if model_path.is_file():
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            print(f"[Self-Play] Modelo cargado desde {model_path}")
        else:
            print(f"[Self-Play] No se encontró {model_path}, se usa modelo sin entrenar.")

    cfg = SelfPlayConfig(
        num_games=args.games,
        num_simulations=args.sims,
        temperature=args.temperature,
        temp_drop_move=args.temp_drop_move,
    )

    generate_self_play_data(
        model=model,
        output_dir=args.output_dir,
        config=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
