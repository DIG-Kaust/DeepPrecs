import numpy as np
import torch
from torch.utils.data import DataLoader


def noise_input(inp, noise_std, device='cpu'):
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


def train_validate(autoencoder, train_dataset, valid_dataset, distance, optimizer,
                   num_epochs, batch_size, add_noise=False, noise_std=0., device='cpu'):
    """Training loop

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
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = {"train": dataloader_train, "val": dataloader_valid}
    datalengths = {"train": len(dataloader_train),
                   "val": len(dataloader_valid)}

    # NN Training
    train_loss_history = np.zeros(num_epochs)
    train_mse_loss_history = np.zeros(num_epochs)
    valid_mse_loss_history = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                # set model to training mode
                autoencoder.train()
            else:
                # set model to evaluate mode
                autoencoder.eval()
            running_mse_loss = 0.0
            running_loss = 0.0
            for data in dataloaders[phase]:
                inp = data[0]
                inp = inp.to(device)
                if add_noise:
                    noise_inp = noise_input(inp, noise_std, device=device)
                else:
                    noise_inp = inp
                # forward
                output, encoded = autoencoder(noise_inp)
                mse_loss = distance(output, inp)
                loss = mse_loss
                # backward + optimize only if in training phase
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    running_mse_loss += mse_loss.item()
                else:
                    loss = mse_loss
                running_loss += loss.item()
            if phase == 'train':
                train_mse_loss_history[epoch] = running_mse_loss / datalengths[phase]
                train_loss_history[epoch] = running_loss / datalengths[phase]
            else:
                valid_mse_loss_history[epoch] = running_loss / datalengths[phase]
        print('epoch [{}/{}] -- train: {:.8f}, Valid: {:.8f}'.format(epoch+1, num_epochs, train_loss_history[epoch],
                                                                     valid_mse_loss_history[epoch]))
    return train_mse_loss_history, valid_mse_loss_history, train_loss_history