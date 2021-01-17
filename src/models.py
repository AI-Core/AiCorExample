import torch


def get(args):
    if args.model == "baseline":
        return Baseline(args)
    else:
        return Serious(args)


# 25%
class Baseline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = torch.nn.Linear(32 * 32, args.num_classes)

    def forward(self, inputs):
        return self.layer(inputs)


# 95% train, 70% on validation
class Serious(torch.nn.Module):
    def __init__(self, args):
        # float, half
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(32 * 32, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, args.num_classes),
        )

    def forward(self, inputs):
        return self.model(inputs)
