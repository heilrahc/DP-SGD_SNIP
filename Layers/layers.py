import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from opacus.grad_sample import register_grad_sampler
from opacus.grad_sample import GradSampleModule
from typing import Dict



class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, name=None):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.name = name
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', name=None):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.name = name
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, name=None):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.name = name
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        if self.affine:
            W = self.weight * self.weight_mask
            b = self.bias * self.bias_mask
        else:
            W = self.weight
            b = self.bias

        return F.group_norm(input, self.num_groups, W, b, self.eps)


@register_grad_sampler(Linear)
def custom_linear_grad_sample(layer: Linear, activations: torch.Tensor, backprops: torch.Tensor) -> Dict[nn.Parameter, torch.Tensor]:
    print("Registering custom Linear grad sampler")
    weight = layer.weight * layer.weight_mask
    bias = layer.bias * layer.bias_mask if layer.bias is not None else None

    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {weight: gs}

    if bias is not None:
        ret[bias] = torch.einsum("n...k->nk", backprops)

    return ret


@register_grad_sampler(Conv2d)
def custom_conv2d_grad_sample(layer: Conv2d, A: torch.Tensor, B: torch.Tensor):
    print("Registering custom Conv2d grad sampler")
    weight = layer.weight * layer.weight_mask
    bias = layer.bias * layer.bias_mask if layer.bias is not None else None

    # Unfold the input tensor
    A_unfold = F.unfold(A, layer.kernel_size, layer.dilation, layer.padding, layer.stride)

    # Calculate the per-sample gradients
    gs = torch.einsum("nij,nijkl->nikl", B, A_unfold)

    # Rearrange the dimensions to match the weight tensor
    gs = gs.reshape(gs.shape[0], *weight.shape)

    ret = {weight: gs}
    if layer.bias is not None:
        ret[bias] = torch.einsum("n...k->nk", B)

    return ret


@register_grad_sampler(GroupNorm)
def custom_groupnorm_grad_sample(layer: GroupNorm, A: torch.Tensor, B: torch.Tensor):
    print("Registering custom GroupNorm grad sampler")
    weight = layer.weight * layer.weight_mask if layer.affine else None
    bias = layer.bias * layer.bias_mask if layer.affine else None

    C = A.shape[1] // layer.num_groups
    norm_shape = (1, layer.num_groups, C, *A.shape[2:])
    A = A.reshape(*norm_shape)
    B = B.reshape(*norm_shape)

    mean = A.mean(dim=(2, 3), keepdim=True)
    A_centered = A - mean
    std = torch.sqrt(A_centered.pow(2).mean(dim=(2, 3), keepdim=True) + layer.eps)

    grad_sample_std = (B * A_centered).sum(dim=(2, 3), keepdim=True) / (std * std)
    grad_sample_mean = B.sum(dim=(2, 3), keepdim=True) - grad_sample_std * A_centered.sum(dim=(2, 3), keepdim=True) / std
    grad_sample = (-A_centered / (std * std) * grad_sample_std - 1 / std * grad_sample_mean) / B.shape[2] / B.shape[3]

    ret = {}
    if layer.affine:
        ret[weight] = torch.einsum("nc...->nc", grad_sample)
        ret[bias] = torch.einsum("nc...->nc", B)

    return ret


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, name=None):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.name=name
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, name=None):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.name = name
        if self.affine:
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W



