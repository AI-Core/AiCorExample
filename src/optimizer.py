import torch


def get(args, parameters):
    return getattr(torch.optim, args.optimizer)(parameters, lr=args.lr)
