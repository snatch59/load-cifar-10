# load-cifar-10

Utility to load cifar-10 image data into training and test data sets.

Download the cifar-10 python version dataset from
[here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), and
extract the cifar-10-batches-py folder into the same directory as the
load_cifar_10.py script.

The code contains example usage, and runs under Python 3 only. Note that
the load_cifar_10_data() function has the option to load the images as
negatives using negatives=True.
