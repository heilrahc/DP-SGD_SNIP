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
from opacus import PrivacyEngine


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


def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    clip = 1
    delta = 1e-5

    privacy_losses = []
    test_losses = []
    acctop1s = []
    acctop5s = []
    sigmas = [5]
    clips = [1, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5]
    # test_losses_e = []
    # acctop1s_e = []
    # acctop5s_e = []
    for sigma in sigmas:
        # for compression in compressions:
        test_losses_k = []
        acctop1s_k = []
        acctop5s_k = []
        privacy_losses_k = []
        for k in range(1):
            privacy_engine = PrivacyEngine()
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
            model = nn.DataParallel(model)
            model = model.to(device)
            #TODO: restore pretrained model's weight(prob not here)
            loss = nn.CrossEntropyLoss()
            opt_class, opt_kwargs = load.optimizer(args.optimizer)
            optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
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
                       clip, sigma, args,
                       args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
            prune_dataset_size = sys.getsizeof([data for idx,(data,target) in enumerate(prune_loader)][0])

            epsilon, _ = privacy_analyze(sigma, delta, 1, 1, prune_dataset_size)
            print("Epsilon: ", epsilon)

            if 0:
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

    #     test_loss = np.mean(test_losses_k)
    #     acc1 = np.mean(acctop1s_k)
    #     acc5 = np.mean(acctop5s_k)
    #     eps = np.mean(privacy_losses_k)
    #     # loss_dev = statistics.stdev(test_losses_k)
    #     # acc1_dev = statistics.stdev(acctop1s_k)
    #     # acc5_dev = statistics.stdev(acctop5s_k)
    #
    #     print("Epsilon: ", eps, "Test Loss: ", test_loss, "Acc1: ", acc1, "Acc5: ", acc5)
    #
    #     test_losses.append(test_loss)
    #     acctop1s.append(acc1)
    #     acctop5s.append(acc5)
    #     privacy_losses.append(eps)
    #     # test_losses_e.append(loss_dev)
    #     # acctop1s_e.append(acc1_dev)
    #     # acctop5s_e.append(acc5_dev)
    #
    #
    # print("Test Loss: ", test_losses)
    # print("Acc1: ", acctop1s)
    # print("Acc5: ", acctop5s)
    # print("Epsilon", privacy_losses)
    # plotGraph(privacy_losses, test_losses, "epsilon", "test_losses", args)
    # plotGraph(privacy_losses, acctop1s, "epsilon", "acctop1s", args)
    # plotGraph(privacy_losses, acctop5s, "epsilon", "acctop5s", args)





def plotGraph(x, y, x_label, y_label, args):
    folder = "Results/plots"
    plt.errorbar(x, y, fmt='-o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    pltsName = args.compression + x_label + y_label + ".png"
    plt.savefig(folder + pltsName)
    plt.clf()


