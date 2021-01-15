import parser

import torch

import data
import metrics
import models
import optimizer
import runner


def main():
    # Add seed
    args = parser.get()

    data_class = data.Dataset(args)
    train, validation = data_class.train(), data_class.validation()

    model = models.get(args)
    optimizer = optimizer.get(args, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_metrics = runner.run(
            model,
            criterion,
            optimizer,
            train,
            True,
            {"loss": metrics.loss, "accuracy": metrics.accuracy},
        )
        metrics.print(train_metrics)
        validation_metrics = runner.run(
            model,
            criterion,
            optimizer,
            validation,
            False,
            {"loss": metrics.loss, "accuracy": metrics.accuracy},
        )
        metrics.print(validation_metrics)


if __name__ == "__main__":
    main()
