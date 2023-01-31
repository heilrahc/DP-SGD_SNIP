import torch

from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.subsampling import subsample


class MaskedLinearBatchGrad(FirstOrderModuleExtension):
    """Extract individual gradients for a MaskedLinear module``."""

    def __init__(self):
        """Store parameters for which individual gradients should be computed."""
        super().__init__(params=["weight", "weight_mask"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for MaskedLinear's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """
        subsampling = ext.get_subsampling()
        batch_axis = 0
        individual_gradients = LinearDerivatives().param_mjp(
            "weight",
            module,
            g_inp,
            g_out,
            subsample(g_out[0], dim=batch_axis, subsampling=subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )
        masked_individual_gradients = individual_gradients * module.weight_mask
        return masked_individual_gradients

    def weight_mask(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for MaskedLinear's ``weight_mask`` parameter."""
        subsampling = ext.get_subsampling()
        batch_axis = 0
        individual_gradients_of_product = LinearDerivatives().param_mjp(
            "weight",
            module,
            g_inp,
            g_out,
            subsample(g_out[0], dim=batch_axis, subsampling=subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )
        individual_gradients = individual_gradients_of_product * module.weight
        return individual_gradients


class MaskedConv2dBatchGrad(FirstOrderModuleExtension):
    """Extract individual gradients for a MaskedLinear module``."""

    def __init__(self):
        """Store parameters for which individual gradients should be computed."""
        super().__init__(params=["weight", "weight_mask"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for MaskedLinear's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """
        subsampling = ext.get_subsampling()
        batch_axis = 0
        individual_gradients = Conv2DDerivatives().param_mjp(
            "weight",
            module,
            g_inp,
            g_out,
            subsample(g_out[0], dim=batch_axis, subsampling=subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )
        masked_individual_gradients = individual_gradients * module.weight_mask
        return masked_individual_gradients

    def weight_mask(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for MaskedLinear's ``weight_mask`` parameter."""
        subsampling = ext.get_subsampling()
        batch_axis = 0
        individual_gradients_of_product = Conv2DDerivatives().param_mjp(
            "weight",
            module,
            g_inp,
            g_out,
            subsample(g_out[0], dim=batch_axis, subsampling=subsampling),
            sum_batch=False,
            subsampling=subsampling,
        )
        individual_gradients = individual_gradients_of_product * module.weight
        return individual_gradients
