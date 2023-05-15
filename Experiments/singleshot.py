import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from backpack import extend
import tensorflow as tf
import statistics
import matplotlib.pyplot as plt
from autodp import rdp_acct, rdp_bank
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
from Layers import layers
from opacus import PrivacyEngine
import os
from opacus.grad_sample import register_grad_sampler
from typing import Dict
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights
from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule


def CGF_func_single(sigma1):
    func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
    func = lambda x: func_gaussian_1(x)

    return func

def privacy_analyze(sigma, delta, epochs, batch, dataset_size):
    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma = sigma  # noise level for privatizing h

    # (2) desired delta level
    delta = delta

    # (3) sampling rate
    n_epochs = epochs  # depending on your experiment length, change the number of epochs for training
    batch_size = batch  # depending on your mini-batch size, change this value

    n_data = dataset_size  # depending on your dataset size, change this value
    steps_per_epoch = n_data // batch_size
    k = steps_per_epoch * n_epochs  # k is the number of steps during the entire training
    prob = batch_size / n_data  # prob is the subsampling probability

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    # define the functional form of uppder bound of RDP
    func = CGF_func_single(sigma)

    eps_seq = []
    print_every_n = 100
    # print("Number of steps: ", k)
    for i in range(1, k + 1):
        acct.compose_subsampled_mechanism(func, prob)
        eps_seq.append(acct.get_eps(delta))
        if i % print_every_n == 0 or i == k:
            print("[", i, "]Privacy loss is", (eps_seq[-1]))

    eps_final = acct.get_eps(delta)
    # print("The final epsilon delta values after the training is over: ", (eps_final, delta))
    #
    # print(f"DP args: epochs: {n_epochs}, batch: {batch_size}, sigma: {sigma}, delta: {delta}")

    return eps_final, delta


def jax_to_torch(params):
    torch_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            torch_params[key] = jax_to_torch(value)
        else:
            torch_params[key] = torch.from_numpy(np.asarray(value))
    return torch_params


def find_weight_key(mask_key):
    if 'softmax' in mask_key.lower():
        return "Softmax"
    elif 'first_conv' in mask_key.lower():
        return 'First_conv'
    elif 'final_norm' in mask_key.lower():
        return 'Final_norm'
    else:
        part = mask_key.split("/")[1]
        if 'skip' in part.lower():
            parts = part.split("_")
            a = parts[1]
            if 'conv' in part.lower():
                return f"Block_{a}.0.skip_conv"
            else:
                return f"Block_{a}.0.skip_norm"
        else:
            parts = part.split("_")
            b = parts[-2]
            c = parts[-1]
            if 'conv' in part.lower():
                a = parts[-3][0] #block number
                return f"Block_{a}.{b}.conv{c}"
            else:
                a = parts[1]
                return f"Block_{a}.{b}.norm{c}"


jax_params = np.load("DP-SGD_SNIP/jax_params.npy", allow_pickle=True).item()
torch_params = jax_to_torch(jax_params)
torch_params = {find_weight_key(key): value for key, value in torch_params.items()}



def reinstantiate(name, param):
    jax_param = torch_params[name]
    if 'w' in jax_param.keys():
        if param.shape == jax_param['w'].squeeze(0).T.shape:
            param = jax_param['w'].squeeze(0).T
    else:
        if param.shape == jax_param['scale'].squeeze(0).T.shape:
            param = jax_param['scale'].squeeze(0).T
    return param

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    clip = 1
    delta = 1e-5

    sigmas = [0]
    # compressions = [0, 0.2, 0.4, 0.8, 0.9, 1.0, 1.2, 1.3, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4]
    # compressions = [0, 0.2, 0.4, 0.8, 0.9, 1.0, 1.2, 1.3]
    compressions = [0.2]
    test_losses_e_comp = []
    acctop1s_e_comp = []
    acctop5s_e_comp = []
    privacy_losses_compression = []
    acc1s_compression = []
    acc5s_compression = []
    test_losses_compression = []
    for compression in compressions:
        # args.compression = compression
        privacy_losses = []
        test_losses = []
        acctop1s = []
        acctop5s = []
        test_losses_e = []
        acctop1s_e = []
        acctop5s_e = []
        for sigma in sigmas:
            test_losses_k = []
            acctop1s_k = []
            acctop5s_k = []
            privacy_losses_k = []
            for k in range(1):
                ## Data ##
                print('Loading {} dataset.'.format(args.dataset))
                input_shape, num_classes = load.dimension(args.dataset)
                prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
                train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
                test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

                ## Model, Loss, Optimizer ##
                print('Creating {}-{} model.'.format(args.model_class, args.model))
                model = load.model(args.model, args.model_class)(input_shape,
                                                                 num_classes,
                                                                 args.dense_classifier,
                                                                 args.pretrained).to(device)
                privacy_engine = PrivacyEngine()

                # # reinitiate the params using jax code's params
                # for name, module in model.named_modules():
                #     for param in module.parameters(recurse=False):
                #         param = reinstantiate(name, param)

                # model = ModuleValidator.fix(model)
                # ModuleValidator.validate(model, strict=False)
                # model = GradSampleModule(model)
                model = nn.DataParallel(model)
                model = model.to(device)
                loss = nn.CrossEntropyLoss()
                opt_class, opt_kwargs = load.optimizer(args.optimizer)
                optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)

                # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                #     module=model, optimizer=optimizer, data_loader=train_loader, target_epsilon=sigma, target_delta=delta, epochs=args.post_epochs, max_grad_norm=clip, grad_sample_mode="hook")


                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

                ## Pre-Train ##
                print('Pre-Train for {} epochs.'.format(args.pre_epochs))
                pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                             test_loader, device, args.pre_epochs, args.verbose)

                ## Prune ##
                print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
                pruner = load.pruner(args.pruner)(
                    generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                # sparsity = 10 ** (-float(args.compression))
                sparsity = 10 ** (-float(args.compression))
                prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                           args.compression_schedule, args.mask_scope, args.prune_epochs,
                           clip, sigma,
                           args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
                prune_dataset_size = sys.getsizeof([data for idx,(data,target) in enumerate(train_loader)][0])

                epsilon, _ = privacy_analyze(sigma, delta, 1, 1, prune_dataset_size)
                print("Epsilon: ", epsilon)

                prune_result = metrics.summary(model,
                                               pruner.scores,
                                               metrics.flop(model, input_shape, device),
                                               lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
                total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
                possible_params = prune_result['size'].sum()
                print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params,
                                                                  total_params / possible_params))
                ## Post-Train ##
                print('Post-Training for {} epochs.'.format(args.post_epochs))
                post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                              test_loader, device, args.post_epochs, args.verbose)

                ## Display Results ##
                frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
                train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
                prune_result = metrics.summary(model,
                                               pruner.scores,
                                               metrics.flop(model, input_shape, device),
                                               lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
                total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
                possible_params = prune_result['size'].sum()
                total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
                possible_flops = prune_result['flops'].sum()


                print("Compression: ", args.compression)
                print("Train results:\n", train_result)
                print("Prune results:\n", prune_result)
                print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params,
                                                                  total_params / possible_params))
                print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
                print("------------------")

                test_losses_k.append(post_result.tail(1)['test_loss'].iloc[0])
                acctop1s_k.append(post_result.tail(1)['top1_accuracy'].iloc[0])
                acctop5s_k.append(post_result.tail(1)['top5_accuracy'].iloc[0])
                privacy_losses_k.append(epsilon)

            test_loss = np.mean(test_losses_k)
            acc1 = np.mean(acctop1s_k)
            acc5 = np.mean(acctop5s_k)
            eps = np.mean(privacy_losses_k)
            loss_dev = statistics.stdev(test_losses_k)
            acc1_dev = statistics.stdev(acctop1s_k)
            acc5_dev = statistics.stdev(acctop5s_k)


            print("Epsilon: ", eps, "Test Loss: ", test_loss, "Acc1: ", acc1, "Acc5: ", acc5)

            test_losses.append(test_loss)
            acctop1s.append(acc1)
            acctop5s.append(acc5)
            privacy_losses.append(eps)
            test_losses_e.append(loss_dev)
            acctop1s_e.append(acc1_dev)
            acctop5s_e.append(acc5_dev)

        print("Compression: -------------------", args.compression)
        print("Test Loss: ", test_losses)
        print("Acc1: ", acctop1s)
        print("Acc5: ", acctop5s)
        print("Epsilon", privacy_losses)
        name = "compression:" + str(args.compression) + "_" + "epsilons" + "_"
        # plotGraph(privacy_losses, test_losses, "epsilon", "test_losses", name + "loss")
        # plotGraph(privacy_losses, acctop1s, "epsilon", "acctop1s", name + "acc1")
        # plotGraph(privacy_losses, acctop5s, "epsilon", "acctop5s", name + "acc5")
        print("---------------------------------------------")

        privacy_losses_compression.append(privacy_losses)
        acc1s_compression.append(acctop1s)
        acc5s_compression.append(acctop5s)
        test_losses_compression.append(test_losses)
        test_losses_e_comp.append(test_losses_e)
        acctop1s_e_comp.append(acctop1s_e)
        acctop5s_e_comp.append(acctop5s_e)

    print("************Summary**********")
    print("Test Loss: ", test_losses_compression)
    print("Test Loss dev: ", test_losses_e_comp)
    print("Acc1: ", acc1s_compression)
    print("Acc1 dev: ", acctop1s_e_comp)
    print("Acc5: ", acc5s_compression)
    print("Acc5 dev: ", acctop5s_e_comp)
    print("Epsilon", privacy_losses_compression)


    # for j in range(len(sigmas)):
    #     accuracies_eps_j = [row[j] for row in acc1s_compression]
    #     epsilon = privacy_losses_compression[0][j]
    #     name = "EPS:" + str(epsilon) + "_" + "compression vs acc1"
    #     plotGraph(compressions, accuracies_eps_j, "compression", "acctop1s", name)




    # print("Test Loss: ", test_losses)
    # print("Acc1: ", acctop1s)
    # print("Acc5: ", acctop5s)
    # print("Epsilon", privacy_losses)
    # plotGraph(privacy_losses, test_losses, "epsilon", "test_losses", args)
    # plotGraph(privacy_losses, acctop1s, "epsilon", "acctop1s", args)
    # plotGraph(privacy_losses, acctop5s, "epsilon", "acctop5s", args)





def plotGraph(x, y, x_label, y_label, name):
    plt.figure()
    plt.xscale("log")
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)

    folder = 'DPSNIP_Results/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    pltsName = name + ".png"
    file_path = os.path.join(folder, pltsName)

    if not os.path.exists(file_path):
        with open(file_path, 'w'):
            pass

    plt.savefig(file_path)


