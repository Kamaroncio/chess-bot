# engine/mcts.py

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import chess

from .config import POLICY_SIZE
from .game import board_to_tensor, move_to_index, index_to_move, game_outcome_to_value


class Node:
    """
    Nodo de MCTS para ajedrez.

    Se guarda:
    - tablero (python-chess.Board)
    - estadísticas N, W, Q
    - prior P
    - hijos: dict[action_index, Node]
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional["Node"],
        prior: float,
    ):
        self.board: chess.Board = board
        self.parent: Optional[Node] = parent
        self.prior: float = prior

        self.children: Dict[int, Node] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0

        # Color que va a mover en este nodo
        self.to_play: bool = board.turn

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    Implementación básica de Monte Carlo Tree Search guiado por red neuronal.

    - model: instancia de ChessNet (o compatible)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_simulations: int = 800,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

    def run(self, board: chess.Board, add_dirichlet_noise: bool = True) -> Tuple[np.ndarray, Node]:
        """
        Ejecuta MCTS desde el tablero dado y devuelve:
        - pi: distribución de probabilidad sobre las 4096 acciones
              (frecuencia de visitas en la raíz).
        - root: nodo raíz de la búsqueda.
        """
        root = Node(board=board.copy(), parent=None, prior=1.0)

        # Expandimos raíz si no lo está aún
        if not root.is_expanded():
            self._expand(root)

        # Ruido de Dirichlet en la raíz para exploración (solo en self-play)
        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node, path = self._select(root)
            value = self._evaluate_and_expand(node)
            self._backpropagate(path, value)

        # Construir distribución pi (visit counts) en la raíz
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = 0

        for action_idx, child in root.children.items():
            pi[action_idx] = child.visit_count
            total_visits += child.visit_count

        if total_visits > 0:
            pi /= total_visits

        return pi, root

    def _add_dirichlet_noise(self, root: Node) -> None:
        """
        Mezcla el prior P con ruido de Dirichlet en la raíz.
        """
        if not root.children:
            return

        action_indices = list(root.children.keys())
        n_actions = len(action_indices)

        noise = np.random.dirichlet([self.dirichlet_alpha] * n_actions)
        for a, n in zip(action_indices, noise):
            child = root.children[a]
            child.prior = child.prior * (1 - self.dirichlet_epsilon) + n * self.dirichlet_epsilon

    def _select(self, root: Node) -> Tuple[Node, List[Node]]:
        """
        Selecciona un nodo hoja a partir de la raíz siguiendo la regla PUCT.
        Devuelve el nodo hoja y el camino (lista de nodos desde raíz hasta hoja).
        """
        node = root
        path = [node]

        while node.is_expanded() and not node.board.is_game_over(claim_draw=True):
            best_score = -float("inf")
            best_action = None
            best_child = None

            # Total de visitas de los hijos
            total_visits = sum(child.visit_count for child in node.children.values())
            sqrt_total = math.sqrt(total_visits + 1e-8)

            for action_idx, child in node.children.items():
                q = child.q_value
                u = self.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
                score = q + u

                if score > best_score:
                    best_score = score
                    best_action = action_idx
                    best_child = child

            if best_child is None:
                break

            node = best_child
            path.append(node)

        return node, path

    def _evaluate_and_expand(self, node: Node) -> float:
        """
        Evalúa el nodo con la red si no está terminal.
        Si el tablero está terminado, devuelve directamente el valor del resultado.
        Si no, expande hijos con priors desde la política de la red.
        Devuelve un valor en [-1,1] desde la perspectiva del jugador que va a mover en node.
        """
        board = node.board

        # Si la partida ha terminado en este nodo, no expandimos
        if board.is_game_over(claim_draw=True):
            result = game_outcome_to_value(board, perspective=node.to_play)
            return result.value

        # Obtenemos tensor del tablero
        state_tensor = board_to_tensor(board)  # (8,8,C)
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0)  # (1,8,8,C)
        state_tensor = state_tensor.to(self.device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            # logits: (1, POLICY_SIZE)
            # value: (1, 1)
            logits = logits[0].cpu().numpy()
            value = float(value[0].item())

        # Softmax para obtener distribución de política completa
        # (sobre las 4096 posibles acciones)
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        policy = exp_logits / (np.sum(exp_logits) + 1e-12)

        # Expansión de hijos solo sobre jugadas legales
        legal_moves = list(board.legal_moves)
        # Suma de priors de jugadas legales (para renormalizar si se quiere)
        legal_priors_sum = 0.0

        children: Dict[int, Node] = {}

        for move in legal_moves:
            action_idx = move_to_index(move)
            prior = float(policy[action_idx])
            legal_priors_sum += prior

            # Generar tablero hijo
            child_board = board.copy()
            child_board.push(move)

            child_node = Node(board=child_board, parent=node, prior=prior)
            children[action_idx] = child_node

        # Renormalizamos priors solo sobre jugadas legales
        if legal_priors_sum > 0:
            for a, child in children.items():
                child.prior = child.prior / legal_priors_sum

        node.children = children

        # value ya está en [-1,1] desde la perspectiva del jugador que mueve en node
        return value

    def _expand(self, node: Node) -> None:
        """
        Expansión inicial de un nodo (igual que _evaluate_and_expand pero ignorando el valor).
        Se usa típicamente solo para la raíz.
        """
        board = node.board

        if board.is_game_over(claim_draw=True):
            return

        state_tensor = board_to_tensor(board)
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0)
        state_tensor = state_tensor.to(self.device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            logits = logits[0].cpu().numpy()

        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        policy = exp_logits / (np.sum(exp_logits) + 1e-12)

        legal_moves = list(board.legal_moves)
        legal_priors_sum = 0.0
        children: Dict[int, Node] = {}

        for move in legal_moves:
            action_idx = move_to_index(move)
            prior = float(policy[action_idx])
            legal_priors_sum += prior

            child_board = board.copy()
            child_board.push(move)

            child_node = Node(board=child_board, parent=node, prior=prior)
            children[action_idx] = child_node

        if legal_priors_sum > 0:
            for a, child in children.items():
                child.prior = child.prior / legal_priors_sum

        node.children = children

    def _backpropagate(self, path: List[Node], value: float) -> None:
        """
        Propaga el valor desde la hoja hasta la raíz.
        value está en la perspectiva del jugador que movía en la hoja.
        Cada vez que subimos un nivel, la perspectiva cambia de signo.
        """
        v = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += v
            # Al subir, cambia la perspectiva (el turno alterna)
            v = -v
