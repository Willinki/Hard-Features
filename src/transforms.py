import torch
from torch import nn


class SignTransform:
    """
    Trasformazione che applica il segno elemento per elemento e converte in float.
    """

    def __call__(self, x):
        return torch.sign(x).float()


class RandomProjectionTransform(nn.Module):
    """
    Trasformazione che proietta casualmente da dim_in a dim_out usando pesi random.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.register_buffer("weight", torch.randn(dim_in, dim_out))

    def forward(self, x):
        assert len(x.shape) == 3, "Input must be a 3D tensor"
        x = x.flatten()
        return torch.matmul(x, self.weight)
