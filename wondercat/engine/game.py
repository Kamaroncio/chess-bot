# engine/game.py

from __future__ import annotations

import numpy as np
import chess
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from .config import BOARD_SIZE, NUM_PLANES, POLICY_SIZE


PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


@dataclass
class GameResult:
    """
    Representa el resultado de la partida.

    result_str: "1-0", "0-1", "1/2-1/2" o "*"
    winner: True (blancas), False (negras), None (tablas o sin ganador)
    value: valor numérico entre -1 y 1 desde la perspectiva del jugador
           que estaba al turno en el estado evaluado.
    """
    result_str: str
    winner: Optional[bool]
    value: float


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convierte un tablero de python-chess a tensor (8,8,14) float32.

    Canales:
    0-5: piezas blancas P,N,B,R,Q,K
    6-11: piezas negras P,N,B,R,Q,K
    12: turno (1 si blancas, 0 si negras)
    13: info extra simple (1 si hay derechos de enroque para alguien)
    """
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PLANES), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        offset = 0 if piece.color == chess.WHITE else 6
        planes[row, col, offset + PIECE_PLANES[piece.piece_type]] = 1.0

    # Turno
    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    # Info simple: si hay algún derecho de enroque
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        planes[:, :, 13] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """
    Codifica una jugada en un índice [0, 4095] usando from*64 + to.

    Nota: solo se distingue por casillas origen/destino.
    Esto implica que para promociones vamos a usar, por diseño,
    la promoción a dama como canónica.
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Decodifica un índice [0, 4095] a una jugada legal en el tablero dado.

    - Si from/to coinciden con una jugada legal sin promoción -> se devuelve.
    - Si es una jugada de promoción, se intenta usar promoción a dama.
    - Si no existe ninguna jugada legal con ese from/to -> None.
    """
    if index < 0 or index >= POLICY_SIZE:
        return None

    from_square = index // 64
    to_square = index % 64

    candidate = chess.Move(from_square, to_square)

    # Caso simple: jugada normal o enroque/en passant ya codificada así
    if candidate in board.legal_moves:
        return candidate

    # Caso promoción: intentamos con promoción a dama
    # python-chess representa promociones con el campo "promotion"
    # Probamos la jugada con promotion=QUEEN
    promo_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    if promo_move in board.legal_moves:
        return promo_move

    # Puede haber otras promociones (torre, alfil, caballo), las ignoramos por simplicidad.
    # Si ninguna cuadra, devolvemos None.
    return None


def is_game_over(board: chess.Board) -> bool:
    return board.is_game_over(claim_draw=True)


def game_outcome_to_value(board: chess.Board, perspective: bool) -> GameResult:
    """
    Convierte el resultado del tablero a valor numérico desde la perspectiva
    de 'perspective' (True=blancas, False=negras).

    perspective es el color del jugador al que le tocaba mover en el estado
    que estamos evaluando (importante para RL).
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # No está realmente terminada la partida (por seguridad).
        # Valor 0 como neutro.
        return GameResult(result_str="*", winner=None, value=0.0)

    result_str = outcome.result()
    winner = outcome.winner  # True blancas, False negras, None tablas

    if winner is None:
        value = 0.0
    else:
        # +1 si gana el jugador 'perspective', -1 si pierde.
        value = 1.0 if winner == perspective else -1.0

    return GameResult(result_str=result_str, winner=winner, value=value)


class ChessGame:
    """
    Pequeño wrapper sobre python-chess.Board para facilitar el uso desde MCTS y entrenamiento.
    """

    def __init__(self, board: Optional[chess.Board] = None):
        self.board: chess.Board = board.copy() if board is not None else chess.Board()

    @property
    def turn(self) -> bool:
        """True si blancas, False si negras."""
        return self.board.turn

    def clone(self) -> "ChessGame":
        return ChessGame(self.board)

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def apply_move(self, move: chess.Move) -> None:
        self.board.push(move)

    def undo_move(self) -> None:
        self.board.pop()

    def is_game_over(self) -> bool:
        return is_game_over(self.board)

    def result(self) -> Optional[GameResult]:
        if not self.is_game_over():
            return None
        # perspective: jugador que tenía el turno en el último estado real
        # Aquí usamos el jugador que está en "turn" tras acabar.
        perspective = self.board.turn
        return game_outcome_to_value(self.board, perspective)

    def to_tensor(self) -> np.ndarray:
        return board_to_tensor(self.board)

    def fen(self) -> str:
        return self.board.fen()

    @staticmethod
    def from_fen(fen: str) -> "ChessGame":
        board = chess.Board(fen=fen)
        return ChessGame(board)
