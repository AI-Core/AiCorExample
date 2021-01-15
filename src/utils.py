import torch


def random_split(data, validation_percent):
    validation_points = int(validation_percent * len(data))
    return torch.utils.data.random_split(
        data, [validation_points, len(data) - validation_points]
    )
