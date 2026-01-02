import torch
from torch import nn


def maybe_flatten_parameters(model: nn.Module, device: str):
    if str(device).startswith("cuda"):
        for m in model.modules():
            if isinstance(m, torch.nn.GRU):
                m.flatten_parameters()
