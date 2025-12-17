# run_training.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from engine.model import ChessNet
from training.trainer import TrainingConfig, train_model


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo de ajedrez con datos de self-play.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/selfplay",
        help="Directorio con ficheros .npz de self-play.",
    )
    parser.add_argument(
        "--model-in",
        type=str,
        default=None,
        help="Modelo inicial (.pt) opcional para continuar entrenamiento.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/chess_model.pt",
        help="Ruta donde guardar el modelo entrenado.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,   # <<< MÁS PEQUEÑO PARA NO CARGAR TANTO LA CPU
        help="Tamaño de batch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay para Adam.",
    )
    parser.add_argument(
        "--policy-weight",
        type=float,
        default=1.0,
        help="Peso de la loss de política.",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="Peso de la loss de valor.",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=1e-4,
        help="Regularización L2 manual adicional.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,    # <<< Nº máximo de hilos de CPU para entrenar
        help="Número máximo de hilos de CPU que puede usar PyTorch.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet()
    model.to(device)

    if args.model_in is not None:
        model_in_path = Path(args.model_in)
        if model_in_path.is_file():
            state = torch.load(model_in_path, map_location=device)
            model.load_state_dict(state)
            print(f"[Train] Modelo inicial cargado desde {model_in_path}")
        else:
            print(f"[Train] No se encontró {model_in_path}, se entrena desde cero.")

    cfg = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_weight,
        value_loss_weight=args.value_weight,
        l2_reg=args.l2_reg,
        num_threads=args.threads,
    )

    stats = train_model(model, cfg)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[Train] Modelo guardado en {out_path}")
    print(f"[Train] Stats: {stats}")


if __name__ == "__main__":
    main()
