import torch


def accuracy(logits, y):
    return torch.mean((torch.argmax(logits, dim=-1) == y).float())


def loss(logits, y):
    return torch.nn.functional.cross_entropy(logits, y)


def print(metrics):
    for name, value in metrics:
        print("Name: ", value)
