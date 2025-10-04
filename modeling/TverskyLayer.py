import torch
import torch.nn as nn
import torch.nn.functional as F

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

        #nn.init.uniform_(self.alpha, 0.0004, 0.012)
        #nn.init.uniform_(self.beta, -0.004, 0.004)
        #nn.init.uniform_(self.theta, -.001, 0.001)

    # Shifted indicator since we need a differentiable signal > 0 but binary mask is not such a thing.
    def indicator(self, x):
        sigma = (torch.tanh(self.approximate_sharpness * x) + 1) * 0.5
        weighted = x * sigma
        return weighted, sigma

    # Ignorematch with Product intersections
    def forward(self, x):
        B = x.size(0)

        # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
        A = x @ self.features.T          # [B, F]
        Pi = self.prototypes @ self.features.T  # [P, F]

        sigma_A, weighted_A = self.indicator(A)      # [B, F]
        sigma_Pi, weighted_Pi = self.indicator(Pi)    # [P, F]

        common = weighted_A @ weighted_Pi.T            # [B,F] @ [F,P] = [B,P]
        distinctive_A = weighted_A @ (1 - sigma_Pi).T  # [B,F] @ [F,P] = [B,P]
        distinctive_B = (1 - sigma_A) @ weighted_Pi.T  # [B,F] @ [F,P] = [B,P]

        # Fix for amp, seems to get confused if we don't specifically cast these.
        alpha = self.alpha.to(common.dtype)
        beta = self.beta.to(common.dtype)
        theta = self.theta.to(common.dtype)

        #  S = θ·C - α·D_A - β·D_B
        return theta * common - alpha * distinctive_A - beta * distinctive_B # [B, P]
