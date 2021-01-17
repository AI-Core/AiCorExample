import torch
import torchvision

import transforms
import utils


class Dataset:
    def __init__(self, args):
        self.args = args
        data = torchvision.datasets.FakeData(
            size=args.size,
            image_size=(1, 32, 32),
            num_classes=args.num_classes,
            transform = torchvision.transforms.ToTensor(),
        )
        self.validation_data, self.train_data = utils.random_split(
            data, args.validation_percent
        )

    def validation(self):
        return torch.utils.data.DataLoader(
            self.validation_data,
            batch_size=self.args.batch_size,
            pin_memory=self.args.device == "cuda",
        )

    def train(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=self.args.device == "cuda",
        )
