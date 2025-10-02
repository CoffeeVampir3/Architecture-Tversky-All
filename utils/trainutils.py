import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time

def count_parameters_layerwise(model):
    # Layerwise params, turn this into a util function.
    total_params = 0
    layer_params = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param_count = parameter.numel()
        layer_params[name] = param_count
        total_params += param_count

    print(f"\nModel Parameter Summary:")
    print("-" * 60)
    for name, count in layer_params.items():
        print(f"{name}: {count:,} parameters")
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params:,}\n")

    return total_params
