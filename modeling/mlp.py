import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.006)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
