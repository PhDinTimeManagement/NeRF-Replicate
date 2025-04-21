"""
Integrator implementing quadrature rule.
"""
from itertools import accumulate
from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.

        # Step 1: Compute alpha_i = 1 - exp(-sigma_i * delta_i)
        alpha = 1.0 - torch.exp(-sigma * delta) # [num_ray, num_sample]

        # Step 2: Compute cumulative transmittance (T_i = exp(- sum sigma_j * delta_j)) using exclusive cumsum
        # Cumulative sum of sigma * delta: [num_ray, num_sample]
        accumulated = torch.cumsum(sigma * delta, dim=-1) # inclusive

        # Shift right and pad with 0 to make it exclusive
        shifted = torch.roll(accumulated, shifts=1, dims=-1)
        shifted[:, 0] = 0.0  # no transmittance before first sample

        # T_i = exp(- cumulative)
        transmittance = torch.exp(-shifted)  # [num_ray, num_sample]

        # Step 3: Compute weights = T_i * alpha_i
        weights = transmittance * alpha # [num_ray, num_sample]

        # Step 4: Final RGB = sum_i weights_i * color_i
        rgbs = torch.sum(weights.unsqueeze(-1) * radiance, dim=-2) # [num_ray, 3]

        return rgbs, weights
