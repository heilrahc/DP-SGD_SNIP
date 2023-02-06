import argparse
# make sure to install autodp in your environment by
# pip install autodp

# Two variants of computing privacy loss:
# (1) privatize_both = 1: when we privatize both h and remaining gradients after pruning based on h computed on private data
# (2) privatize_both = 0: when we privatize only the remaining gradients after pruning based on h computed on public data

from autodp import rdp_acct, rdp_bank


# get the CGF functions
def CGF_func(sigma1, sigma2):
    func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
    func_gaussian_2 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma2}, x)

    func = lambda x: func_gaussian_1(x) + func_gaussian_2(x)

    return func


def CGF_func_single(sigma1):
    func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
    func = lambda x: func_gaussian_1(x)

    return func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma1", default=1., type=float)
    parser.add_argument("--sigma2", default=1., type=float)
    parser.add_argument("--C", default=0.3, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--nodes1", default=50, type=int)
    parser.add_argument("--nodes2", default=100, type=int)
    parser.add_argument("--clipnorm", default=1e-5, type=float)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--n_training_data", default=60000, type=int)  # this is true if we use the MNIST data.
    args = parser.parse_args()
    return args


def main(privatize_both, args):
    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma1 = args.sigma1  # noise level for privatizing h
    sigma2 = args.sigma2  # noise level for privatising remaining gradients after pruning

    # (2) desired delta level
    delta = 1e-5

    # (3) sampling rate
    n_epochs = args.epochs  # depending on your experiment length, change the number of epochs for training
    batch_size = args.batch  # depending on your mini-batch size, change this value

    n_data = args.n_training_data  # depending on your dataset size, change this value
    steps_per_epoch = n_data // batch_size
    k = steps_per_epoch * n_epochs  # k is the number of steps during the entire training
    prob = batch_size / n_data  # prob is the subsampling probability

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    # define the functional form of uppder bound of RDP
    if privatize_both == 1:
        func = CGF_func(sigma1, sigma2)  # we redefine CFG for double Gaussian mechanisms
    else:
        func = CGF_func_single(sigma2)

    eps_seq = []
    print_every_n = 100
    print("Number of steps: ", k)
    for i in range(1, k + 1):
        acct.compose_subsampled_mechanism(func, prob)
        eps_seq.append(acct.get_eps(delta))
        if i % print_every_n == 0 or i == k:
            print("[", i, "]Privacy loss is", (eps_seq[-1]))

    eps_final = acct.get_eps(delta)
    print("The final epsilon delta values after the training is over: ", (eps_final, delta))
    if privatize_both == 1:
        print(f"DP args: epochs: {n_epochs}, batch: {batch_size}, sigma1: {sigma1}, sigma2: {sigma2}, delta: {delta}")
    else:
        print(f"DP args: epochs: {n_epochs}, batch: {batch_size}, sigma2: {sigma2}, delta: {delta}")

    return eps_final, delta


if __name__ == '__main__':
    privatize_both = 1
    args = parse_args()
    print(args)
    eps_final, delta_final = main(privatize_both, args)

    filename = "priv_results.csv"
    file = open(filename, "a+")
    file.write(f"{args.sigma1}, {args.sigma2}, {args.epochs}, {args.batch}, {eps_final} \n")
    file.close()


