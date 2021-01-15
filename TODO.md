## Learning rate schedulers

- As outlined in BatchNorm notebook, after some training epochs passed loss stales
or even increases.
- Check [Adjusting Learning Rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
in PyTorch for possible fixes. Code schedulers module which allows us to choose
learning rate appropriately

## Automatic Mixed Precision

- If you want to speed up your training you may want to adjust this code
to use [AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)

## Transformations

- Add choose-able from cmdline transformations
- Apply them __only__ on training dataset (see [torchdata](https://github.com/szymonmaszke/torchdata))
to specify them easier.

## Settings

- The more settings we add, the harder it is to handle them using command line
- Using settings file (like `.yaml`) might help. See [hydra](https://github.com/facebookresearch/hydra)
for one possibility to further refactor the code
