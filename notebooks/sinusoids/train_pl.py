import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule, Callback


def noise_input(inp, noise_std):
    """Add noise to input

    Parameters
    ----------
    inp : :obj:`torch.tensor`
        Input
    noise_std : :obj:`float`
        Noise standard deviation

    Returns
    -------
    out : :obj:`torch.tensor`
        Noisy input

    """
    return inp + torch.randn(inp.size(), dtype=inp.dtype).type_as(inp) * noise_std


class DataModule(LightningDataModule):
    """Create dataloaders

    Parameters
    ----------
    x_train : :obj:`numpy.ndarray`
        Training data of size `nsamples x nt`
    x_valid : :obj:`numpy.ndarray`
        Validation data of size `nsamples x nt`
    batch_size : :obj:`int`, optional
        Batch size

    """
    def __init__(self, x_train, x_valid, batch_size=32):
        super().__init__()
        self.x_train = x_train
        self.x_valid = x_valid
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
    network : :obj:`torch.Module`
        Autoencoder network
    criterion : :obj:`torch.nn.Module`
        Criterion to use in loss
    noise_std : :obj:`float`
        Noise standard deviation to add to inputs as a way
        of regularizing the training process

    """
    def __init__(self, network, criterion, noise_std=0.):
        super().__init__()
        self.network = network
        self.criterion = criterion
        self.noise_std = noise_std

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Single training step
        """
        inp= batch[0]

        #add noise
        if  self.noise_std > 0.:
            noise_inp = noise_input(inp, self.noise_std)
        else:
            noise_inp = inp

        # forward
        output, encoded = self.network(noise_inp)
        loss = self.criterion(output, inp)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """Single validation step
        """
        inp = val_batch[0]

        # add noise
        if self.noise_std > 0.:
            noise_inp = noise_input(inp, self.noise_std)
        else:
            noise_inp = inp

        # forward
        output, encoded = self.network(noise_inp)
        loss = self.criterion(output, inp)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback
    """

    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.valid_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.valid_loss.append(trainer.callback_metrics["val_loss"].item())