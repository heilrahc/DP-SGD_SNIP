import torch

def custom_conv2d_grad_sample(module: torch.nn.Module, A: torch.Tensor, B: torch.Tensor):
    # A: input to the layer, shape (batch_size, in_channels, H_in, W_in)
    # B: output gradient, shape (batch_size, out_channels, H_out, W_out)

    # In your specific case, you can use the default Conv2d sampler provided by Opacus
    from opacus.grad_sample import _compute_conv_grad_sample

    # Get the masked weight and bias from your custom Conv2d layer
    weight = module.weight * module.weight_mask
    bias = module.bias * module.bias_mask if module.bias is not None else None

    # Compute per-sample gradients using the default Conv2d sampler
    grad_sample = _compute_conv_grad_sample(A, B, weight, bias, module.stride, module.padding, module.dilation, module.groups)

    return grad_sample


def custom_groupnorm_grad_sample(module: torch.nn.Module, A: torch.Tensor, B: torch.Tensor):
    # A: input to the layer, shape (batch_size, num_channels, H, W)
    # B: output gradient, shape (batch_size, num_channels, H, W)

    # In this specific case, you can use the default GroupNorm sampler provided by Opacus
    from opacus.grad_sample import _compute_groupnorm_grad_sample

    # Get the masked weight and bias from your custom GroupNorm layer
    weight = module.weight * module.weight_mask if module.affine else None
    bias = module.bias * module.bias_mask if module.affine else None

    # Compute per-sample gradients using the default GroupNorm sampler
    grad_sample = _compute_groupnorm_grad_sample(A, B, weight, bias, module.num_groups, module.eps)

    return grad_sample


def custom_linear_grad_sample(module: torch.nn.Module, A: torch.Tensor, B: torch.Tensor):
    # A: input to the layer, shape (batch_size, in_features)
    # B: output gradient, shape (batch_size, out_features)

    # In this specific case, you can use the default Linear sampler provided by Opacus
    from opacus.grad_sample import _compute_linear_grad_sample

    # Get the masked weight and bias from your custom Linear layer
    weight = module.weight * module.weight_mask
    bias = module.bias * module.bias_mask if module.bias is not None else None

    # Compute per-sample gradients using the default Linear sampler
    grad_sample = _compute_linear_grad_sample(A, B, weight, bias)

    return grad_sample
