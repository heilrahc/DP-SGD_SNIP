from tqdm import tqdm
import torch
import numpy as np


def get_sparsity(tensor):
    total_elements = tensor.size
    zero_elements = np.count_nonzero(tensor == 0)
    sparsity_ratio = zero_elements / total_elements
    return sparsity_ratio


def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs, clip, noise, args,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device, clip, noise)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        mask = pruner.mask(sparse, scope)
        param = pruner.param()

        # for k, v in param:
        #     print(k, v.shape)
        mask_np = {k: v.cpu().numpy() for k, v in mask}
        param_np = {k: v.cpu().detach().numpy() for k, v in param}
        for k, v in mask_np.items():
            print(k, get_sparsity(v))

        sparsity2 = (1 - 1 / pow(10, float(args.compression)))
        np.savez(f"jax_privacy/pruned_torch_weights_{args.model}_{args.dataset}_{args.pruner}_{sparsity2:.3f}.npz", **mask_np)
        np.savez("jax_privacy/jax_privacy/pruned_torch_weights.npz", **mask_np)
        np.savez("jax_privacy/torch_params.npz", **param_np)
        np.savez("jax_privacy/jax_privacy/torch_params.npz", **param_np)
        print("---------pruned weights stored----------")

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()
