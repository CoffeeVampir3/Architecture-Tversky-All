import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TverskyLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        prototypes: int | nn.Parameter,
        features: int | nn.Parameter,
        prototype_init=None,
        feature_init=None,
        approximate_sharpness=13
    ):
        super().__init__()

        if isinstance(features, int):
            self.features = nn.Parameter(torch.empty(features, input_dim))
        elif isinstance(features, nn.Parameter):
            self.features = features
        else:
            raise ValueError("features must be int or nn.Parameter")

        if isinstance(prototypes, int):
            self.prototypes = nn.Parameter(torch.empty(prototypes, input_dim))
        elif isinstance(prototypes, nn.Parameter):
            self.prototypes = prototypes
        else:
            raise ValueError("prototypes must be int or nn.Parameter")

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.theta = nn.Parameter(torch.zeros(1))

        self.prototype_init = prototype_init
        self.feature_init = feature_init
        self.approximate_sharpness = approximate_sharpness

        self.reset_parameters()

    def reset_parameters(self):
        if self.feature_init is not None:
            self.feature_init(self.features)
        if self.prototype_init is not None:
            self.prototype_init(self.prototypes)

        # From the paper:
        # We observed that α > β in trained Tversky neural networks...
        # So we might as well help them out.
        nn.init.uniform_(self.alpha, 0.004, 0.12)
        nn.init.uniform_(self.beta, 0.001, 0.004)
        nn.init.uniform_(self.theta, 0.002, 0.06)

    # Shifted indicator since we need a differentiable signal > 0 but binary mask is not such a thing.
    def indicator(self, x):
        return (torch.tanh(self.approximate_sharpness * x) + 1) / 2

    # Ignorematch with Product intersections
    def forward(self, x):
        B = x.size(0)

        # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
        A = x @ self.features.T          # [B, F]
        Pi = self.prototypes @ self.features.T  # [P, F]

        sigma_A = self.indicator(A)      # [B, F]
        sigma_Pi = self.indicator(Pi)    # [P, F]

        weighted_A = A * sigma_A         # [B, F]
        weighted_Pi = Pi * sigma_Pi      # [P, F]

        common = weighted_A @ weighted_Pi.T            # [B,F] @ [F,P] = [B,P]

        # This is an approximation that will not distinguish highly similar features but
        # is actually tractable for networks larger than a breadbox.
        distinctive_A = weighted_A @ (1 - sigma_Pi).T  # [B,F] @ [F,P] = [B,P]
        distinctive_B = (1 - sigma_A) @ weighted_Pi.T  # [B,F] @ [F,P] = [B,P]

        #  S = θ·C - α·D_A - β·D_B
        return self.theta * common - self.alpha * distinctive_A - self.beta * distinctive_B # [B, P]
