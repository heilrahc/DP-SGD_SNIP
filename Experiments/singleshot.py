import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from backpack import extend
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

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
    loss = nn.CrossEntropyLoss()
    model = extend(model)
    loss = extend(loss)
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    clip_norms = np.linspace(1e-7, 1, 5)
    noise_factors = np.linspace(1e-3, 1e3, 5)
    best_clip = [0, 0, 0]
    best_noise = [0, 0, 0]
    test_errors = []
    acctop1s = []
    acctop5s = []
    best_test_error = float("inf")
    best_acctop1 = float("-inf")
    best_acctop5 = float("-inf")

    for clip in clip_norms:
        test_errors_row = []
        acctop1s_row = []
        acctop5s_row = []
        for noise in noise_factors:
            ## Pre-Train ##
            print('Pre-Train for {} epochs.'.format(args.pre_epochs))
            pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                         test_loader, device, args.pre_epochs, args.verbose)

            ## Prune ##
            print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
            pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            sparsity = 10**(-float(args.compression))
            prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                       args.compression_schedule, args.mask_scope, args.prune_epochs,
                       clip, noise,
                       args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)


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
            test_errors_row.append(post_result.tail(1)['test_loss'].iloc[0])
            acctop1s_row.append(post_result.tail(1)['top1_accuracy'].iloc[0])
            acctop5s_row.append(post_result.tail(1)['top5_accuracy'].iloc[0])

            if post_result.tail(1)['test_loss'].iloc[0] < best_test_error:
                best_test_error = post_result.tail(1)['test_loss'].iloc[0]
                best_clip[0] = clip
                best_noise[0] = noise
            if post_result.tail(1)['top1_accuracy'].iloc[0] > best_acctop1:
                best_acctop1 = post_result.tail(1)['top1_accuracy'].iloc[0]
                best_clip[1] = clip
                best_noise[1] = noise
            if post_result.tail(1)['top5_accuracy'].iloc[0] > best_acctop5:
                best_acctop5 = post_result.tail(1)['top5_accuracy'].iloc[0]
                best_clip[2] = clip
                best_noise[2] = noise

            # print("Train results:\n", train_result)
            # print("Prune results:\n", prune_result)
            # print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
            # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
        test_errors.append(test_errors_row)
        acctop1s.append(acctop1s_row)
        acctop5s.append(acctop1s_row)

    print("best clips: ", best_clip)
    print("best_noises: ", best_noise)
    print("best_test_error: ", best_test_error)
    print("best_acctop1: ", best_acctop1)
    print("best_acctop5: ", best_acctop5)
    plotGraph(clip_norms, test_errors[:][1], "clips", "test_errors")
    plotGraph(clip_norms, acctop1s[:][1], "clips", "acctop1")
    plotGraph(clip_norms, acctop5s[:][1], "clips", "acctop5")
    plotGraph(noise_factors, test_errors[1], "noises", "test_errors")
    plotGraph(noise_factors, acctop1s[1], "noises", "acctop1")
    plotGraph(noise_factors, acctop5s[1], "noises", "acctop5")





    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

def plotGraph(x, y, x_label, y_label):
    folder = "Results/plots"
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    pltsName = x_label + y_label + ".png"
    plt.savefig(folder + pltsName)
    plt.clf()