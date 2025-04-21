"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO

        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim

        # First 4 layers (before skip connection)
        self.fc_pos = nn.ModuleList([
            nn.Linear(pos_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        ])

        # Layers after concatentation γ(x) again (skip connection)
        self.fc_pos_skip = nn.ModuleList([
            nn.Linear(feat_dim + pos_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        ])

        # Sigma head (predict density)
        self.fc_sigma = nn.Linear(feat_dim, 1)

        # Feature Layer before color head
        self.fc_feat = nn.Linear(feat_dim, feat_dim)

        # View direction layer after concatentation γ(d) (skip connection)
        self.fc_view = nn.Linear([feat_dim + view_dir_dim, 128])

        # RGB head (predict color)
        self.fc_rgb = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO

        x = pos
        for i in range(4):
            x = self.relu(self.fc_pos[i](x))

        # Skip connection: concatenate input pos again
        x = torch.cat([x, pos], dim=-1)

        for i in range(4):
            x = self.relu(self.fc_pos_skip[i](x))

        # Sigma (density) prediction
        sigma = self.relu(self.fc_sigma(x))

        # Feature for RGB head
        feat = self.fc_feat(x)

        # Concatenate with view direction
        h = torch.cat([feat, view_dir], dim=-1)
        h = self.relu(self.fc_view(h))

        # RGB prediction
        rgb = self.sigmoid(self.fc_rgb(h))

        return sigma, rgb