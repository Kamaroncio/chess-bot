# training/__init__.py

from .self_play import SelfPlayConfig, generate_self_play_data
from .dataset import ChessDataset, load_data
from .trainer import TrainingConfig, train_model
