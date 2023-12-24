from pathlib import Path

import torch
from torch import nn, optim

MODEL_STATE_DICT_KEY = "model_state_dict"
OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
EPOCH_KEY = "epoch"


def save_model_and_optimizer(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, filepath: Path
):
    torch.save(
        {
            EPOCH_KEY: epoch,
            MODEL_STATE_DICT_KEY: model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
        },
        filepath,
    )
