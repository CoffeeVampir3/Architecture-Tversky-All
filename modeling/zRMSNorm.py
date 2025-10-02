import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = norm * (1.0 + self.weight.float())
        return output - output.mean(-1, keepdim=True)
