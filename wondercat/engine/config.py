BOARD_SIZE = 8  # 8x8
NUM_PLANES = 14  # 6 blancas + 6 negras + turno + info extra simple
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 64*64 = 4096

# Valor objetivo de las partidas:
# -1 derrota, 0 tablas, +1 victoria (desde la perspectiva del jugador al que le tocaba mover).
VALUE_MIN = -1.0
VALUE_MAX = 1.0
