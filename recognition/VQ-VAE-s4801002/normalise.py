import torch

class Normalise(object):
    """
    Normalise an input tensor's values to the range 0-1 by dividing by the maximum value in the tensor.
    """

    def __call__(self, inputs):
        max_val = torch.max(inputs)
        return inputs / max_val