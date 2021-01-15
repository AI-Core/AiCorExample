import argparse


def get():
    parser = argparse.ArgumentParser(description="Some classification task.")
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for neural network"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    # Optimizer use choices argument
    parser.add_argument("--optimizer", default="SGD", help="Optimizer")
    parser.add_argument(
        "--model",
        choices=["baseline", "serious"],
        default="baseline",
        help="Model to choose from",
    )

    # Check between [0,1]
    parser.add_argument(
        "--validation_percent", type=float, default=0.2, help="Optimizer"
    )

    parser.add_argument(
        "--num_classes", type=int, default=10, help="Classes for classification"
    )

    parser.add_argument(
        "--size", type=int, default=5000, help="Classes for classification"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device to use for training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    get()
