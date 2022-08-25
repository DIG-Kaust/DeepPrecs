import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM, MS_SSIM
from torchvision.utils import make_grid


def pearsonr(x, y):
    """ Pearson Correlation Coefficient

    Pearson Correlation Coefficient between two inputs, ranging from -1 to 1.
    from https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py

    Parameters
    ----------
    x : :obj:`torch.tensor`
        First input
    y : :obj:`torch.tensor`
        Second input

    """
    dim = -1
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    corr = bessel_corrected_covariance / (x_std * y_std + 1e-10)
    return corr


def loss_pearson(x, y):
    """Pearson loss

    Pearson Correlation Coefficient rescaled as 1 - R to span the range from 0 to 2.

    Parameters
    ----------
    x : :obj:`torch.tensor`
        First input
    y : :obj:`torch.tensor`
        Second input

    """
    r = pearsonr(x, y)
    b, c, h, w = r.size()
    return (torch.ones_like(r) - r).div(b * c * h)


def loss_corr(x, y):
    """ Concordance Correlation Coefficient (CCC)

    Concordance Correlation Coefficient (CCC) scaled as 1 - CCC to
    span the range from 0 to 2.

    Parameters
    ----------
    x : :obj:`torch.tensor`
        First input
    y : :obj:`torch.tensor`
        Second input

    """
    dim = -1
    bessel_correction_term = (x.shape[dim] - 1) / x.shape[dim]
    r = pearsonr(x, y)
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    ccc = 2 * r * x_std * y_std / (x_std * x_std
                                   + y_std * y_std
                                   + (x_mean - y_mean)
                                   * (x_mean - y_mean)
                                   / bessel_correction_term + 1e-10)
    b, c, h, w = ccc.size()
    return (torch.ones_like(ccc) - ccc).div(b * c * h)


class MS_SSIM_Loss(MS_SSIM):
    """Multi-scale SSIM Loss

    Multi-scale SSIM rescaled as 1-MS_SSIM to span the range from 0 to 2.

    Parameters
    ----------
    x : :obj:`torch.tensor`
        First input
    y : :obj:`torch.tensor`
        Second input

    """
    def forward(self, x, y):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(x, y) )


class SSIM_Loss(SSIM):
    """SSIM Loss

    SSIM rescaled as 1-MS_SSIM to span the range from 0 to 2.

    Parameters
    ----------
    x : :obj:`torch.tensor`
        First input
    y : :obj:`torch.tensor`
        Second input

    """
    def forward(self, x, y):
        return 100*( 1 - super(SSIM_Loss, self).forward(x, y) )


class WeightedMSE(nn.MSELoss):
    """Weighted MSE Loss

    Parameters
    ----------
    weight : :obj:`torch.tensor`
        Weights

    """
    def __init__(self, weight):
        super().__init__(reduction='none')
        self.weights = weight
    def forward(self, input, target, weight=None):
        if weight is not None:
            self.weights = weight.clone()
        out = self.weights * super().forward(input, target)
        return torch.mean(out)



