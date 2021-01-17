import collections
import metrics
import torch

# Train
# Validate
# On given arguments, data


def run(model, criterion, optimizer, dataset, is_training: bool, metrics):
    model.train(is_training)

    dictionary = collections.defaultdict(int)

    counter = 0
    with torch.set_grad_enabled(is_training):
        for X, y in dataset:
            counter += 1
            y_pred = model(X.squeeze().reshape(X.shape[0], -1))

            loss = criterion(y_pred, y)
            for name, metric in metrics.items():
                dictionary[name] += metric(y_pred, y)

            if is_training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    return {name: value / counter for name, value in dictionary.items()}
