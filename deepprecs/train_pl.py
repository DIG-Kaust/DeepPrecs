import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule, Callback
from sklearn.model_selection import train_test_split
from deepprecs.utils import EarlyStopping, noise_input, SSIM_Loss, \
    WeightedMSE, loss_corr, loss_pearson, show_tensor_images

lossfuncs = {'mse': nn.MSELoss(),
             'weightmse': WeightedMSE(None),
             'l1': nn.L1Loss(),
             'ssim': SSIM_Loss(win_size=7, win_sigma=0.5, data_range=1.,
                               size_average=True, channel=1),
             'pearson': loss_pearson,
             'mse_pearson': [nn.MSELoss(), loss_pearson],
             'l1_pearson': [nn.L1Loss(), loss_pearson],
             'ccc': loss_corr,
             'mse_ccc': [nn.MSELoss(), loss_corr],
             'l1_ccc': [nn.L1Loss(), loss_corr]}


class DataModule(LightningDataModule):
    """Create dataloaders

    Parameters
    ----------
    xs : :obj:`numpy.ndarray`
        Set of data of size `ns x nt`

    """
    def __init__(self, xs, valid_size=0.1, random_state=42, batch_size=32):
        super().__init__()
        x_train, x_valid, _, _ = train_test_split(xs, xs, test_size=valid_size,
                                                  random_state=random_state)
        self.x_train = x_train
        self.x_valid = x_valid
        # force to have all batches of same size
        # self.x_valid = x_valid[:(x_valid.shape[0] // batch_size) * batch_size]
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = TensorDataset(torch.from_numpy(self.x_train.astype(np.float32)),)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = TensorDataset(torch.from_numpy(self.x_valid.astype(np.float32)),)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class LitAutoencoder(pl.LightningModule):
    """Pytorch-Lightning AutoEncoder module

    Parameters
    ----------
    nh : :obj:`int`
        Height of input images
    nw : :obj:`torch.nn.Module`
        Width of input images
    nenc : :obj:`int`
        Size of latent code
    network : :obj:`torch.Module`
        Autoencoder network
    lossfunc : :obj:`str`
        Loss function: mse, weightmse, l1, ssim, pearson,
        mse_pearson, l1_pearson, ccc, mse_ccc, l1_ccc
    learning_rate : :obj:`float`, optional
        Learning rate of Adam optimizer
    weight_decay : :obj:`float`, optional
        Weight decay of Adam optimizer
    betas : :obj:`tuple`, optional
        Running average weights for Adam optimizer
    adapt_learning : :obj:`bool`, optional
        Use adaptive learning rate
    lr_scheduler : :obj:`str`, optional
        Learning rate scheduler name (must be `OnPlateau` or `OneCycle`)
    lr_factor : :obj:`tuple`, optional
        Factor by which the learning rate is reduced
    lr_thresh : :obj:`tuple`, optional
        Threshold for measuring the new optimum.
    lr_patience : :obj:`tuple`, optional
        Number of epochs with no improvement after which learning rate will be reduced
    lr_max : :obj:`tuple`, optional
        Upper learning rate boundaries in the cycle for each parameter group
    noise_std : :obj:`float`, optional
        Noise standard deviation to add to inputs as a way
        of regularizing the training process
    noise_std : :obj:`float`, optional
        Percentage of input traces to be masked
#   device : :obj:`str`, optional
#         Device

    """
    def __init__(self, nh, nw, nenc, network, lossfunc, num_epochs, lossweights=None,
                 learning_rate=1e-4, weight_decay=1e-5, betas=(0.9, 0.999),
                 adapt_learning=True, lr_scheduler='OnPlateau', lr_factor=0.1,
                 lr_thresh=0.0001, lr_patience=10, lr_max=0.1,
                 noise_std=0., mask_perc=0., device='cpu'):
        super().__init__()

        # sizes 
        self.nh, self.nw = nh, nw
        self.nenc = nenc
        
        # network
        self.network = network
        
        # training
        self.num_epochs = num_epochs
        self.lossfuncname = lossfunc
        self.lossfunc = lossfuncs[lossfunc]
        self.lossweights = lossweights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.adapt_learning = adapt_learning
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_thresh = lr_thresh
        self.lr_patience = lr_patience
        self.lr_max = lr_max

        # data augumentation
        self.noise_std = noise_std
        self.mask_perc = mask_perc
        self.add_noise = True if noise_std > 0. else False
        self.add_mask = True if mask_perc > 0. else False

        # check if multi-term loss is used
        self.learn_lossweights = False
        if isinstance(self.lossfunc, list):
            self.losses_names = self.lossfuncname.split('_')
            if self.lossweights is None:
                # use learned weights
                self.learn_lossweights = True
                self.lossweights = [torch.zeros((1,)).to(device).detach().requires_grad_(True) for
                                    _ in range(len(self.losses_names))]

        # initialize weights
        self.init_weights(seed=5)

    def _init_weights(self, model):
        if type(model) == nn.Linear:
            torch.nn.init.xavier_uniform(model.weight)
            if model.bias is not None:
                model.bias.data.fill_(0.01)
        elif type(model) == nn.Conv2d or type(model) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform(model.weight)
            if model.bias is not None:
                model.bias.data.fill_(0.01)

    def init_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.network.apply(self._init_weights)

    def configure_optimizer(self):
        if not self.learn_lossweights:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         betas=self.betas,
                                         weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(list(self.parameters()) + self.lossweights,
                                         lr=self.learning_rate,
                                         betas=self.betas,
                                         weight_decay=self.weight_decay)
        return optimizer

    def configure_scheduler(self, optimizer):
        # learning rate update scheduler
        # (see for good summary https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling)
        if self.adapt_learning:
            if self.lr_scheduler == 'OnPlateau':
                scheduler = \
                    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=self.lr_factor,
                                                               threshold=self.lr_thresh,
                                                               patience=self.lr_patience)
            elif self.lr_scheduler == 'OneCycle':
                scheduler = \
                    torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.lr_max,
                                                        cycle_momentum=False,
                                                        epochs=self.num_epochs,
                                                        steps_per_epoch=1)
            else:
                # Assume a scheduler object is passed, if not it will break later
                scheduler = self.lr_scheduler
            return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        lr_scheduler = self.configure_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def mask(self, inp):
        """Apply random mask to input
        """
        imask = np.random.permutation(np.arange(self.nh))[:int(self.nh * self.mask_perc)]
        inp[:, :, imask] = 0
        return inp

    def training_step(self, batch, batch_idx):
        """Single training step
        """
        if isinstance(self.lossfunc, list):
            losses = {}

        # get inputs
        inp = batch[0]
        inp = inp.view(inp.size(0), 1, self.nh, self.nw)

        # add noise
        if self.add_noise:
            noise_inp = noise_input(inp, self.noise_std, self.device)
        else:
            noise_inp = inp.clone()
        if self.add_mask:
            noise_inp = self.mask(noise_inp)

        # forward
        output = self.network(noise_inp)

        # cost function
        if self.lossfuncname == 'ssim':
            loss = self.lossfunc(
                (output + self.mod_normalize) / (2. * self.mod_normalize),
                (inp + self.mod_normalize) / (2. * self.mod_normalize))
        elif self.lossfuncname == 'weightmse':
            weight = torch.std(inp, dim=(1, 2, 3))
            weight = weight / weight.max() * 0.4 + 0.6
            weight = torch.outer(weight, torch.ones(self.nh * self.nw))
            weight = weight.reshape(inp.shape)
            loss = self.lossfunc(output, inp, weight)
        elif self.lossfuncname in ['mse_pearson', 'l1_pearson', 'mse_ccc', 'l1_ccc']:
            losses[self.losses_names[0]] = self.lossfunc[0](output, inp).sum()
            if not self.learn_lossweights:
                loss = self.lossweights[0] * losses[self.losses_names[0]]
            else:
                loss = (torch.exp(-self.lossweights[0]) / 2) * \
                       losses[self.losses_names[0]] + 0.5 * self.lossweights[0]
            for iloss, loss_name in enumerate(self.losses_names[1:]):
                losses[loss_name] = self.lossfunc[iloss + 1](output, inp).sum()
                if not self.learn_lossweights:
                    loss += self.lossweights[iloss + 1] * losses[loss_name]
                else:
                    loss += (torch.exp(-self.lossweights[iloss + 1]) / 2) * \
                            losses[loss_name] + 0.5 * self.lossweights[iloss + 1]
        else:
            loss = self.lossfunc(output, inp)

        # Tracking epoch loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=False)
        if isinstance(self.lossfunc, list):
            for loss_name in self.losses_names:
                self.log(f"train_{loss_name}", losses[loss_name], on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """Single validation step
        """
        """Single training step
                """
        if isinstance(self.lossfunc, list):
            losses = {}

        # get inputs
        inp = val_batch[0]
        inp = inp.view(inp.size(0), 1, self.nh, self.nw)

        # add noise
        if self.add_noise:
            noise_inp = noise_input(inp, self.noise_std, self.device)
        else:
            noise_inp = inp.clone()
        if self.add_mask:
            noise_inp = self.mask(noise_inp)

        # forward
        output = self.network(noise_inp)

        # cost function
        if self.lossfuncname == 'ssim':
            loss = self.lossfunc(
                (output + self.mod_normalize) / (2. * self.mod_normalize),
                (inp + self.mod_normalize) / (2. * self.mod_normalize))
        elif self.lossfuncname == 'weightmse':
            weight = torch.std(inp, dim=(1, 2, 3))
            weight = weight / weight.max() * 0.4 + 0.6
            weight = torch.outer(weight, torch.ones(self.nh * self.nw))
            weight = weight.reshape(inp.shape)
            loss = self.lossfunc(output, inp, weight)
        elif self.lossfuncname in ['mse_pearson', 'l1_pearson', 'mse_ccc', 'l1_ccc']:
            losses[self.losses_names[0]] = self.lossfunc[0](output, inp).sum()
            if not self.learn_lossweights:
                loss = self.lossweights[0] * losses[self.losses_names[0]]
            else:
                loss = (torch.exp(-self.lossweights[0]) / 2) * \
                       losses[self.losses_names[0]] + 0.5 * self.lossweights[0]
            for iloss, loss_name in enumerate(self.losses_names[1:]):
                losses[loss_name] = self.lossfunc[iloss + 1](output, inp).sum()
                if not self.learn_lossweights:
                    loss += self.lossweights[iloss + 1] * losses[loss_name]
                else:
                    loss += (torch.exp(-self.lossweights[iloss + 1]) / 2) * \
                            losses[loss_name] + 0.5 * self.lossweights[iloss + 1]
        else:
            loss = self.lossfunc(output, inp)

        # Store inputs and outputs of first batch for plotting
        if batch_idx == 0:
            self.inp, self.out = inp, output

        # Tracking epoch loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=False)
        if isinstance(self.lossfunc, list):
            for loss_name in self.losses_names:
                self.log(f"val_{loss_name}", losses[loss_name], on_epoch=True, prog_bar=True, logger=False)
        return loss


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback

    Parameters
    ----------
    loss : :obj:`str`
        Loss name (use same convention as in ``LitAutoencoder``)

    """
    def __init__(self, loss=None):
        super().__init__()
        self.train_loss = []
        self.valid_loss = []
        self.multiloss = False
        if loss is not None:
            self.losses_names = loss.split('_')
            if len(self.losses_names) > 1:
                self.multiloss = True
                self.train_losses = {}
                self.valid_losses = {}
                for loss_name in self.losses_names:
                    self.train_losses[loss_name] = []
                    self.valid_losses[loss_name] = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())
        if self.multiloss:
            for loss_name in self.losses_names:
                self.train_losses[loss_name].append(trainer.callback_metrics[f"train_{loss_name}_epoch"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.valid_loss.append(trainer.callback_metrics["val_loss"].item())
        if self.multiloss:
            for loss_name in self.losses_names:
                self.valid_losses[loss_name].append(trainer.callback_metrics[f"val_{loss_name}"].item())


class PlottingCallback(Callback):
    """PyTorch Lightning plotting callback

    Parameters
    ----------
    figdir : :obj:`str`
        Directory path where to save figures

    """

    def __init__(self, figdir):
        super().__init__()
        self.figdir = figdir
        self.epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch > 0:
            inp = trainer.model.inp
            out = trainer.model.out

            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            show_tensor_images(inp.transpose(3, 2))
            plt.title("True (Epoch %d)" % self.epoch)
            plt.subplot(1, 2, 2)
            show_tensor_images(out.transpose(3, 2))
            plt.title("Reconstructed (Epoch %d)" % self.epoch)
            if self.figdir is not None:
                plt.savefig(os.path.join(self.figdir, 'valid_rec_epoch%d.png' %
                                         self.epoch), bbox_inches='tight')
            plt.show()
        self.epoch +=1