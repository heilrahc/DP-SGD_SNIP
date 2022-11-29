import torch
import numpy as np

from backpack import backpack
from backpack.extensions import BatchGrad, BatchL2Grad


class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally."""
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _global_mask_str(self, sparsity):
        """
        removes 3d filters only
        """
        # flatten
        channels_all = []
        for i, (mask, param) in enumerate(self.masked_parameters):
            score = self.scores[id(param)]
            if i != len(self.masked_parameters) - 1:  # outputs all but last
                if len(param.shape) == 4:
                    s = score.mean(-1).mean(-1).mean(-1).flatten()
                    channels_all.append(s)
                if len(param.shape) == 2:
                    s = score.mean(-1).mean(-1).flatten()
                    channels_all.append(s)
        channels_all = torch.cat(channels_all)
        k = int((1.0 - sparsity) * channels_all.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(channels_all, k)  # kth smallest value
            for i, (mask, param) in enumerate(self.masked_parameters):
                print(param.shape)
                score = self.scores[id(param)]
                # if i != 0:  # inputs all but first

                # mask.copy_(torch.where(score <= threshold, zero, one))

                if i != len(self.masked_parameters) - 1:  # outputs all but last
                    if len(param.shape) == 4:
                        s = score.mean(-1).mean(-1).mean(-1).flatten()
                        removed = torch.where(s <= threshold)
                        mask[removed] = 0
                    if len(param.shape) == 2:
                        s = score.mean(-1).mean(-1).flatten()
                        removed = torch.where(s <= threshold)
                        mask[removed] = 0
                    print(removed)

    def _local_mask_str(self, sparsity):
        """
        removes 3d filters only
        """
        # flatten
        for i, (mask, param) in enumerate(self.masked_parameters):
            score = self.scores[id(param)]

            if i != 0:  # inputs all but first
                if i == len(self.masked_parameters) - 1:  # conv to fc
                    # channel_size = int(mask.shape[1]/channel_num)
                    mask_resized = mask.reshape(mask.shape[0], channel_num, -1)
                    mask_resized[:, removed[0]] = 0
                    mask = mask_resized.reshape((mask.shape[0]), -1)
                else:

                    if len(mask.shape) == 4 and not (
                        (mask.shape[0] != mask.shape[1]) and mask.shape[2] == 1
                    ):
                        mask[:, removed[0]] = 0
                    else:
                        mask[:, removed_input[0]] = 0
                    if (mask.shape[0] != mask.shape[1]) and mask.shape[2] != 1:
                        removed_input = removed

            if i != len(self.masked_parameters) - 1:  # outputs all but last
                print(param.shape)
                # get the mean of an output channels for that layer
                # we shouldnt sum/do i for shorcuts just take removed from the previus one
                if len(param.shape) == 4:
                    if not (
                        (mask.shape[0] != mask.shape[1]) and mask.shape[2] == 1
                    ):  # shortcut
                        s = score.mean(-1).mean(-1).mean(-1).flatten()
                if len(param.shape) == 2:
                    s = score.mean(-1).flatten()
                # the kth smallest value of the channel and get indices smaller to remove
                k = int((1.0 - sparsity) * s.numel())
                if not k < 1:
                    # shortcut
                    if len(mask.shape) == 4 and not (
                        (mask.shape[0] != mask.shape[1]) and mask.shape[2] == 1
                    ):
                        threshold, _ = torch.kthvalue(s, k)  # kth smallest value
                        removed = torch.where(s <= threshold)
                    else:
                        removed = removed_shortcut
                    if mask.shape[0] != mask.shape[1]:
                        removed_shortcut = removed
                    # print(removed) # they also are used to remove the inputs to the next layer
                    channel_num = len(s)
                    mask[removed] = 0
                    # mask.copy_(torch.where(score <= threshold, zero, one))
                print(removed)

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise."""
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope."""
        if scope == "global":
            self._global_mask(sparsity)
        if scope == "global_str":
            self._global_mask_str(sparsity)
        if scope == "local":
            self._local_mask(sparsity)
        if scope == "local_str":
            self._local_mask_str(sparsity)
        print("Scope: ", scope)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters."""
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters."""
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


class Change(Pruner):
    def __init__(self, masked_parameters):
        super(Change, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.zeros_like(p)

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

            for m, p in self.masked_parameters:
                grads = torch.clone(p.grad).detach().abs_()
                grads_flatten = grads.flatten()
                grads_flatten_sorted = torch.sort(grads_flatten)
                ind = int(0.5 * grads_flatten_sorted[0].numel())
                grads[torch.where(grads < grads_flatten_sorted[0][ind])] = 0
                grads[torch.where(grads >= grads_flatten_sorted[0][ind])] = 1
                if p.shape[1] == 784:
                    la = 3
                self.scores[id(p)] += grads
                p.grad.zero_()
                # m.grad.zero_()

            end = 0


def expand_vector(vec, tgt_tensor):
    tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
    return vec.view(*tgt_shape)


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18


class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SNIP_DPSGD(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, clip_norm, noise_factor, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            with backpack(BatchGrad(), BatchL2Grad()):
                loss(output, target).backward()

        # first we get all the squared parameter norms...
        squared_param_norms = [m.batch_l2 for m, p in self.masked_parameters]
        # ...then compute the global norms...
        global_norms = torch.sqrt(torch.sum(torch.stack(squared_param_norms), dim=0))
        # ...and finally get a vector of clipping factors
        global_clips = torch.clamp_max(clip_norm / global_norms, 1.0)

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            clipped_sample_grads = m.grad_batch * expand_vector(
                global_clips, m.grad_batch
            )
            # after clipping we sum over the batch
            clipped_grad = torch.sum(clipped_sample_grads, dim=0)
            # gaussian noise standard dev is computed (sensitivity is 2*clip)...
            noise_sdev = noise_factor * 2 * clip_norm
            perturbed_grad = (
                clipped_grad
                + torch.randn_like(clipped_grad, device=device) * noise_sdev
            )  # ...and applied
            m.grad = perturbed_grad  # now we set the parameter gradient to what we just computed

            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=False
            )
            # all parameters in the network
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph (same result for flatten_grads)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=True
            )
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        # , dtype=torch.float64).to(device)
        input = torch.ones([1] + input_dim).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)
