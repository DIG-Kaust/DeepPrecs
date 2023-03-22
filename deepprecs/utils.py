import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM, MS_SSIM
from torchvision.utils import make_grid


def noise_input(inp, noise_std, device):
    """Add noise to input

    Parameters
    ----------
    inp : :obj:`torch.tensor`
        Input
    noise_std : :obj:`float`
        Noise standard deviation
    device : :obj:`str`, optional
        Device

    Returns
    -------
    out : :obj:`torch.tensor`
        Noisy input

    """
    return inp + torch.randn(inp.size(), dtype=inp.dtype).to(device) * noise_std


def show_tensor_images(image_tensor, num_images=25, vmin=-1, vmax=1, cmap='gray'):
    """Visualize images

    Display a batch of images together in a 2D grid

    Parameters
    ----------
    image_tensor : :obj:`torch.tensor`
        Batch of images
    num_images : :obj:`int`, optional
        Number of images to display
    vmin : :obj:`float`, optional
        Min value to display
    vmin : :obj:`float`, optional
        Max value to display
    cmap : :obj:`str`, optional
        Colormap

    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5, normalize=False)
    plt.axis('off')
    plt.imshow(image_grid[0].squeeze(), cmap, vmin=vmin, vmax=vmax)
    plt.axis('tight')