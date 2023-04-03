import torch as pt
import numpy as np
from backpack import backpack
from backpack.extensions import BatchGrad, BatchL2Grad


def dp_sgd_backward(params, loss, device, clip_norm, noise_factor):
  """

  :param params:
  :param loss: computed loss. Must allow sample-wise gradients
  :param device: cpu/gpu key on which the model is run
  :param clip_norm:
  :param noise_factor:
  :return:
  """
  if not isinstance(params, list):
    params = [p for p in params if not isinstance(p, str)]

  #with backpack(BatchGrad(), BatchL2Grad()):
  loss.backward()

  squared_param_norms = [pt.norm(p.grad) for p in params]  # first we get all the squared parameter norms...
  global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))  # ...then compute the global norms...
  global_clips = pt.clamp_max(clip_norm / global_norms, 1.)  # ...and finally get a vector of clipping factors

  for idx, param in enumerate(params):
    #clipped_sample_grads = param.grad_batch * expand_vector(global_clips, param.grad_batch)
    clipped_sample_grads = param.grad * expand_vector(global_clips, param.grad)
    #clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch
    clipped_grad = clipped_sample_grads

    noise_sdev = noise_factor * 2 * clip_norm  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
    perturbed_grad = clipped_grad + pt.randn_like(clipped_grad, device=device) * noise_sdev  # ...and applied
    param.grad = perturbed_grad  # now we set the parameter gradient to what we just computed

  return global_norms, global_clips


def expand_vector(vec, tgt_tensor):
  tgt_shape = [1] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)


def dp_sgd_backward_debug(params, loss, device, clip_norm, noise_factor, debug=True, terminate_after_iteration=True):
  """
  same as above, but optionally prints messages to check a few things:
  - clipped gradients have global norm <= clip-norm
  - empirical standard deviation of noise matches the target one
  - L2 distance between noisy gradient and true gradient
  :param model:
  :param loss: computed loss. Must allow sample-wise gradients
  :param device: cpu/gpu key on which the model is run
  :param clip_norm:
  :param noise_factor:
  :param debug: If true, run asserts
  :param terminate_after_iteration: in debug mode, you likely only want to run one iteration
  :return:
  """
  if not isinstance(params, list):
    params = [p for p in params]

  with backpack(BatchGrad(), BatchL2Grad()):
    loss.backward()

  squared_param_norms = [p.batch_l2 for p in params]  # first we get all the squared parameter norms...
  global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))  # ...then compute the global norms...
  global_clips = pt.clamp_max(clip_norm / global_norms, 1.)  # ...and finally get a vector of clipping factors

  post_clip_norms = []
  for idx, param in enumerate(params):
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips, param.grad_batch)
    clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch

    if debug:
      post_clip_norm = pt.norm(clipped_sample_grads.reshape(clipped_sample_grads.shape[0], -1), dim=1)
      post_clip_norms.append(post_clip_norm)

    noise_sdev = noise_factor * 2 * clip_norm  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
    dp_noise = pt.randn_like(clipped_grad, device=device) * noise_sdev

    if debug:
      print(f'empirical standard deviation of added noise {pt.std(dp_noise)}')

    perturbed_grad = clipped_grad + dp_noise  # ...and applied

    if debug:
      print(f'L2 dist of noisy grad from true one: {pt.norm(param.grad - perturbed_grad)}')

    param.grad = perturbed_grad  # now we set the parameter gradient to what we just computed

  assert idx + 1 == len(squared_param_norms)

  if debug:
    post_clip_global_norms = pt.sqrt(pt.sum(pt.stack([k**2 for k in post_clip_norms]), dim=0))
    print(f'Post clip global norms: min {pt.min(post_clip_global_norms)}, max {pt.max(post_clip_global_norms)}')

    if terminate_after_iteration:
      assert 1 % 2 == 3, 'after debug, the run is stopped.'

  return global_norms, global_clips
