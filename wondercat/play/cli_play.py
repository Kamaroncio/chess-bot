# play/cli_play.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import chess

from engine import ChessGame
from engine.mcts import MCTS
from engine.model import ChessNet
from engine.game import index_to_move
from engine.config import POLICY_SIZE


def load_model(model_path: Optional[str | Path] = None, device: Optional[torch.device] = None) -> ChessNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet()
    model.to(device)

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.is_file():
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            print(f"[CLI] Modelo cargado desde {model_path}")
        else:
            print(f"[CLI] No se encontró {model_path}, se usa modelo sin entrenar.")
    else:
        print("[CLI] No se proporcionó modelo, se usa modelo sin entrenar.")

    return model


def choose_move_with_mcts(
    model: ChessNet,
    board: chess.Board,
    num_simulations: int = 400,
    device: Optional[torch.device] = None,
) -> chess.Move:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        cpuct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        device=device,
    )

    pi, root = mcts.run(board, add_dirichlet_noise=False)

    # Elegimos la jugada con mayor probabilidad
    best_action = int(np.argmax(pi))
    move = index_to_move(best_action, board)

    if move is None or move not in board.legal_moves:
        # Fallback: jugada más visitada de MCTS
        if root.children:
            best_action = max(
                root.children.keys(),
                key=lambda a_idx: root.children[a_idx].visit_count,
            )
            move = index_to_move(best_action, board)

    if move is None or move not in board.legal_moves:
        # Fallback final: primera jugada legal
        move = next(iter(board.legal_moves))

    return move


def print_board(board: chess.Board) -> None:
    board_str = str(board)
    rows = board_str.split("\n")

    files_header = "    a   b   c   d   e   f   g   h"
    separator = "  +---+---+---+---+---+---+---+---+"

    print()
    print(files_header)
    print(separator)

    for r_idx, row in enumerate(rows):
        rank = 8 - r_idx
        cells = row.split(" ")
        # Cada celda es una pieza o '.' para casilla vacía
        print(f"{rank} | " + " | ".join(cells) + f" | {rank}")
        print(separator)

    print(files_header)
    print()
    print(f"FEN: {board.fen()}")
    print()

def ask_human_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    legal_uci = [m.uci() for m in legal_moves]

    print("Tus jugadas legales (UCI):")
    print(", ".join(legal_uci))
    print()

    while True:
        user_input = input("Introduce tu jugada en formato UCI (jugada: ").strip()
        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Formato incorrecto. Intenta de nuevo.")
            continue

        if move in board.legal_moves:
            return move
        else:
            print("Jugada ilegal. Intenta de nuevo.")


def main():
    parser = argparse.ArgumentParser(description="Jugar al ajedrez contra la IA en consola.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al archivo .pt del modelo entrenado (opcional).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=400,
        help="Número de simulaciones de MCTS por jugada de la IA.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=device)

    game = ChessGame()
    board = game.board

    print("Colores: 'w' = juegas con blancas, 'b' = juegas con negras.")
    while True:
        color = input("Elige tu color [w/b]: ").strip().lower()
        if color in ("w", "b"):
            break
        print("Opción no válida.")

    human_is_white = (color == "w")

    print()
    print("Comienza la partida.")
    print("Tablero inicial:")
    print_board(board)

    while not game.is_game_over():
        if board.turn == chess.WHITE and human_is_white:
            print("Tu turno (BLANCAS).")
            move = ask_human_move(board)
            game.apply_move(move)
        elif board.turn == chess.BLACK and not human_is_white:
            print("Tu turno (NEGRAS).")
            move = ask_human_move(board)
            game.apply_move(move)
        else:
            print("Turno de la IA...")
            move = choose_move_with_mcts(model, board, num_simulations=args.sims, device=device)
            print(f"La IA juega: {move.uci()}")
            game.apply_move(move)

        board = game.board
        print_board(board)

    print("Partida terminada.")
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("Resultado desconocido.")
    else:
        print(f"Resultado: {outcome.result()}")
        if outcome.winner is True:
            print("Ganan las BLANCAS.")
        elif outcome.winner is False:
            print("Ganan las NEGRAS.")
        else:
            print("Tablas.")


if __name__ == "__main__":
    main()
