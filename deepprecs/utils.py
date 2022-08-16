import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM, MS_SSIM
from torchvision.utils import make_grid


def set_seed(seed=0):
    """Set the seed of random.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


class EarlyStopping(object):

    def __init__(self, patience=10, max=False, min_delta=0, percentage=False):
        """Stop the optimization when the metrics don't improve.
        Use the `step` method for raising a True to stop.

        https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

        Arguments:
            patience: number of iterations to wait if the stopping condition is met
            max: maximize the metrics instead of minimize
            min_delta: minimum difference between the best and the actual metrics values to trigger the stopping condition
            percentage: min_delta is provided as a percentage of the best metrics value
        """
        self.mode = 'max' if max else 'min'
        self.min_delta = min_delta
        self.patience = patience
        self.percentage = percentage
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better()
        self.msg = "\nEarly stopping called, terminating..."

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            print("Metrics is NaN, terminating...")
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(self.msg)
            return True

        return False

    def _init_is_better(self):
        if self.mode not in {'min', 'max'}:
            raise ValueError('mode ' + self.mode + ' is unknown!')
        if not self.percentage:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - self.min_delta
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + self.min_delta
        else:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - (best * self.min_delta / 100)
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + (best * self.min_delta / 100)


class MS_SSIM_Loss(MS_SSIM):
    """Multi-scale SSIM Loss (1-MS_SSIM to be able to optimize by minimization)
    """
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )


class SSIM_Loss(SSIM):
    """SSIM Loss (1-SSIM to be able to optimize by minimization)
    """
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )


class WeightedMSE(nn.MSELoss):
    def __init__(self, weight):
        super().__init__(reduction='none')
        self.weights = weight
    def forward(self, input, target, weight=None):
        if weight is not None:
            self.weights = weight.clone()
        out = self.weights * super().forward(input, target)
        return torch.mean(out)


def pearsonr(x, y):
    """ Pearson Correlation Coefficient, ranging from -1 to 1.
    from https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py
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
    """ Pearson loss defined as 1 - R to span the range from 0 to 2.
    """
    r = pearsonr(x, y)
    b, c, h, w = r.size()
    return (torch.ones_like(r) - r).div(b * c * h)


def loss_corr(x, y):
    """ Concordance Correlation Coefficient (CCC), ranging from -1 to 1.
    However, as a loss term it becomes 1 - CCC to span the range from 0 to 2.
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


def noise_input(inp, noise_std, device):
    """Add noise to input
    """
    return inp + torch.randn(inp.size(), dtype=inp.dtype).to(device) * noise_std


def show_tensor_images(image_tensor, num_images=25, vmin=-1, vmax=1):
    """Visualizing images

    Given a tensor of images, number of images, an size per image,
    plots and prints the images in an uniform grid.
    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5, normalize=False)
    plt.axis('off')
    plt.imshow(image_grid[0].squeeze(), cmap='gray', vmin=vmin, vmax=vmax)

