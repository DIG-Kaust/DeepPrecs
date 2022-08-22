import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from torch.utils.data import TensorDataset, DataLoader
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

class Train():
    """Train class
    """
    def __init__(self, device, # device
                 nh, nw, nenc, # sizes
                 model, # model
                 lossfunc, weight_decay, learning_rate, # optimizer
                 num_epochs, batch_size, # training
                 mod_normalize=1., # normalization factor for ssim lossfunc
                 lossweights=None, # loss weights for mixed losses
                 noise_std=0., mask_perc=0., # noise or mask to input
                 superres=False, # superresolution (subsample output of model along spatial direction)
                 betas=(0.9, 0.999), # running average weights for Adam,
                 adapt_learning=True, lr_scheduler='OnPlateau', lr_factor=0.1,
                 lr_thresh=0.0001, lr_patience=10, lr_max=0.1, # adaptive learning rat
                 early_stop=True, es_patience=10, es_min_delta=0, # early stop
                 plotflag=False, figdir=None # plotting
                 ):
        # sizes 
        self.nh, self.nw = nh, nw
        self.nenc = nenc

        # model
        self.model = model
        self.superres = superres

        # training
        self.lossfuncname = lossfunc
        self.lossfunc = lossfuncs[lossfunc]
        self.lossweights = lossweights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mod_normalize = mod_normalize
        self.noise_std = noise_std
        self.mask_perc = mask_perc
        self.add_mask = True if mask_perc > 0. else False
        self.add_noise = True if noise_std > 0. else False
        self.device = device

        # check if multi-term loss is used
        self.learn_lossweights = False
        if isinstance(self.lossfunc, list):
            self.losses_names = self.lossfuncname.split('_')
            if self.lossweights is None:
                # use learned weights
                self.learn_lossweights = True
                #self.lossweights = [torch.ones((1,), requires_grad=False) for
                #                    _ in range(len(self.losses_names))]
                #self.lossweights = [w * 0 for w in self.lossweights]
                #self.lossweights = [w.clone().to(self.device).detach().requires_grad_(True) for
                #                    w in self.lossweights]
                self.lossweights = [torch.zeros((1,)).to(self.device).detach().requires_grad_(True) for
                                    _ in range(len(self.losses_names))]
                #self.lossweights = [(torch.rand((1,)) /10.).to(self.device).detach().requires_grad_(True) for
                #                    _ in range(len(self.losses_names))]

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        if not self.learn_lossweights:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate,
                                              betas=self.betas,
                                              weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(list(self.model.parameters()) + self.lossweights,
                                              lr=self.learning_rate,
                                              betas=self.betas,
                                              weight_decay=self.weight_decay)
        # learning rate update scheduler
        # (see for good summary https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling)
        self.adapt_learning = adapt_learning
        self.lr_scheduler = lr_scheduler
        if self.adapt_learning:
            if self.lr_scheduler == 'OnPlateau':
                self.scheduler = \
                    torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               mode='min',
                                                               factor=lr_factor,
                                                               threshold=lr_thresh,
                                                               patience=lr_patience)
            elif self.lr_scheduler == 'OneCycle':
                self.scheduler = \
                    torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                        max_lr=lr_max,
                                                        cycle_momentum=False,
                                                        epochs=self.num_epochs,
                                                        steps_per_epoch=1)
            else:
                # Assume a scheduler object is passed, if not it will break later
                self.scheduler = self.lr_scheduler
        # early stopping (stop after no improvements greater
        # than a certain percentage of the previous loss)
        self.early_stop = early_stop
        if self.early_stop:
            self.stopper = EarlyStopping(patience=es_patience,
                                         min_delta=es_min_delta,
                                         percentage=True)

        # plotting during training
        self.plotflag = plotflag
        self.figdir = figdir

    def load_data(self, xs, valid_size=0.2, random_state=42,
                  shuffle=False, masks=None):
        """Load data (and masks)
        """
        self.use_mask = False if masks is None else True
        if masks is None:
            x_train, x_valid, _, _ = train_test_split(xs, xs, test_size=valid_size,
                                                      random_state=random_state)

            t_train = TensorDataset(torch.from_numpy(x_train.astype(np.float32)),
                                    torch.from_numpy(x_train.astype(np.float32)))
            t_valid = TensorDataset(torch.from_numpy(x_valid.astype(np.float32)),
                                    torch.from_numpy(x_valid.astype(np.float32)))
        else:
            x_train, x_valid, mask_train, mask_valid = \
                train_test_split(xs, masks, test_size=valid_size,
                                 random_state=random_state)
            t_train = TensorDataset(torch.from_numpy(x_train.astype(np.float32)),
                                    torch.from_numpy(mask_train.astype(np.float32)))
            t_valid = TensorDataset(torch.from_numpy(x_valid.astype(np.float32)),
                                    torch.from_numpy(mask_valid.astype(np.float32)))

        dataloader_train = DataLoader(t_train, batch_size=self.batch_size, shuffle=shuffle)
        dataloader_valid = DataLoader(t_valid, batch_size=self.batch_size)
        self.dataloaders = {"train": dataloader_train, "val": dataloader_valid}
        self.datalengths = {"train": len(dataloader_train), "val": len(dataloader_valid)}

        if self.lr_scheduler == 'OneCycle':
            self.scheduler.steps_per_epoch = self.datalengths["train"]
        return x_train, x_valid

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
        self.model.apply(self._init_weights)

    def mask(self, inp):
        imask = np.random.permutation(np.arange(self.nh))[:int(self.nh * self.mask_perc)]
        inp[:, :, imask] = 0
        return inp

    def _train_epoch(self):
        """Train model for one epoch
        """
        # set model to training mode
        self.model.train()

        # train over all samples
        running_loss = 0.0
        if isinstance(self.lossfunc, list):
            losses = {}
            running_losses = {loss_name:0.0 for loss_name in self.losses_names}
        for data in tqdm(self.dataloaders['train']):
            inp, mask = data
            inp = inp.view(inp.size(0), 1, self.nh, self.nw)
            inp = inp.to(self.device)
            if self.use_mask:
                mask = mask.view(inp.size(0), 1, self.nh, self.nw)
                mask = mask.to(self.device)
            # add noise
            if self.add_noise:
                noise_inp = noise_input(inp, self.noise_std, self.device)
            else:
                noise_inp = inp.clone()
            if self.add_mask:
                noise_inp = self.mask(noise_inp)
            # forward
            output = self.model(noise_inp)
            # apply mask when masks are used
            if self.use_mask:
                inp_full = noise_inp.clone()
                output_full = output.clone()
                #inp = inp * mask
                #output = output * mask
                inp = inp[mask.type(torch.ByteTensor)].view(
                    int(mask[:, :, :, 0].sum()), 1, 1, self.nw)
                output = output[mask.type(torch.ByteTensor)].view(
                    int(mask[:, :, :, 0].sum()), 1, 1, self.nw)

            # subsampling for superres
            if self.superres:
                output = output[:, :, ::2, :]
            # cost function
            if self.lossfuncname == 'ssim':
                loss = self.lossfunc(
                    (output + self.mod_normalize) / (2. * self.mod_normalize),
                    (inp + self.mod_normalize) / (2. * self.mod_normalize))
            elif self.lossfuncname == 'weightmse':
                weight = torch.std(inp, dim=(1, 2, 3))
                #weight = (weight / weight.sum()) * self.batch_size
                weight = weight / weight.max() * 0.4 + 0.6
                weight = torch.outer(weight, torch.ones(self.nh * self.nw).to(self.device))
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
                    losses[loss_name] = self.lossfunc[iloss+1](output, inp).sum()
                    if not self.learn_lossweights:
                        loss += self.lossweights[iloss+1] * losses[loss_name]
                    else:
                        loss += (torch.exp(-self.lossweights[iloss+1]) / 2) * \
                                losses[loss_name] + 0.5 * self.lossweights[iloss+1]
            else:
                loss = self.lossfunc(output, inp)
            # backward + optimize
            self.optimizer.zero_grad()
            loss.backward()
            # update the weights
            self.optimizer.step()
            # save losses
            running_loss += loss.item()
            if isinstance(self.lossfunc, list):
                for loss_name in self.losses_names:
                    running_losses[loss_name] += losses[loss_name].item()
        epoch_loss = running_loss / self.datalengths['train']
        if isinstance(self.lossfunc, list):
            epoch_losses = [running_losses[loss_name] / self.datalengths['train']
                            for loss_name in self.losses_names]
        if self.plotflag:
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            show_tensor_images(inp.transpose(3, 2) if not self.use_mask
                               else inp_full.transpose(3, 2))
            plt.title("True (Epoch %d)" % self.epoch)
            plt.subplot(1, 2, 2)
            show_tensor_images(output.transpose(3, 2) if not self.use_mask
                               else output_full.transpose(3, 2))
            plt.title("Reconstructed (Epoch %d)" % self.epoch)
            if self.figdir is not None:
                plt.savefig(os.path.join(self.figdir, 'training_rec_epoch%d.png' %
                                         self.epoch), bbox_inches='tight')
            plt.show()
        if isinstance(self.lossfunc, list):
            return epoch_loss, epoch_losses
        else:
            return epoch_loss

    def _valid_epoch(self):
        """Evaluate model on validation set
        """
        # set model to evaluation mode
        self.model.eval()

        # train over all samples
        running_loss = 0.0
        if isinstance(self.lossfunc, list):
            losses = {}
            running_losses = {loss_name: 0.0 for loss_name in self.losses_names}
        with torch.no_grad():
            for data in self.dataloaders['val']:
                inp, mask = data
                inp = inp.view(inp.size(0), 1, self.nh, self.nw)
                inp = inp.to(self.device)
                if self.use_mask:
                    mask = mask.view(inp.size(0), 1, self.nh, self.nw)
                    mask = mask.to(self.device)
                # add noise
                if self.add_noise:
                    noise_inp = noise_input(inp, self.noise_std, self.device)
                else:
                    noise_inp = inp
                # forward
                output = self.model(noise_inp)
                # extract only non-masked traced when masks are used
                if self.use_mask:
                    # inp = inp * mask
                    # output = output * mask
                    inp = inp[mask.type(torch.ByteTensor)].view(
                        int(mask[:, :, :, 0].sum()), 1, 1, self.nw)
                    output = output[mask.type(torch.ByteTensor)].view(
                        int(mask[:, :, :, 0].sum()), 1, 1, self.nw)
                # subsampling for superres
                if self.superres:
                    output = output[:, :, ::2, :]
                # cost function
                if self.lossfuncname == 'ssim':
                    loss = self.lossfunc((output + self.mod_normalize) / (2. * self.mod_normalize),
                                         (inp + self.mod_normalize) / (2. * self.mod_normalize))
                elif self.lossfuncname == 'weightmse':
                    weight = torch.std(inp, dim=(1, 2, 3))
                    #weight = (weight / weight.sum()) * self.batch_size
                    weight = weight / weight.max() * 0.4 + 0.6
                    weight = torch.outer(weight, torch.ones(self.nh * self.nw).to(self.device))
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
                running_loss += loss.item()
                if isinstance(self.lossfunc, list):
                    for loss_name in self.losses_names:
                        running_losses[loss_name] += losses[loss_name].item()
        epoch_loss = running_loss / self.datalengths['val']
        if isinstance(self.lossfunc, list):
            epoch_losses = [running_losses[loss_name] / self.datalengths['val']
                            for loss_name in self.losses_names]
        if isinstance(self.lossfunc, list):
            return epoch_loss, epoch_losses
        else:
            return epoch_loss

    def train(self):
        """Train model
        """
        self.train_loss_history = torch.zeros(self.num_epochs)
        self.valid_loss_history = torch.zeros(self.num_epochs)
        self.lr_history = torch.zeros(self.num_epochs)
        if isinstance(self.lossfunc, list):
            self.train_losses_history = {lossname:torch.zeros(self.num_epochs) for lossname in self.losses_names}
            self.valid_losses_history = {lossname:torch.zeros(self.num_epochs) for lossname in self.losses_names}
        if self.learn_lossweights:
            self.lossweights_history = {lossname:torch.zeros(self.num_epochs) for lossname in self.losses_names}

        for self.epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            valid_loss = self._valid_epoch()
            # save losses
            if not isinstance(self.lossfunc, list):
                self.train_loss_history[self.epoch] = train_loss
                self.valid_loss_history[self.epoch] = valid_loss
            else:
                self.train_loss_history[self.epoch] = train_loss[0]
                self.valid_loss_history[self.epoch] = valid_loss[0]
                for iloss, lossname in enumerate(self.losses_names):
                    self.train_losses_history[lossname][self.epoch] = \
                        train_loss[1][iloss]
                    self.valid_losses_history[lossname][self.epoch] = \
                        valid_loss[1][iloss]
            # save learning rate
            self.lr_history[self.epoch] = self.optimizer.param_groups[0]["lr"]
            # save weights
            if self.learn_lossweights:
                for iloss, lossname in enumerate(self.losses_names):
                    self.lossweights_history[lossname][self.epoch] = np.exp(-self.lossweights[iloss].item()) / 2

            if self.adapt_learning:
                # adapt learning rate
                if self.lr_scheduler == 'OnPlateau':
                    self.scheduler.step(self.valid_loss_history[self.epoch])
                elif self.lr_scheduler == 'OneCycle':
                    self.scheduler.step()

            if self.early_stop:
                # early stopping
                if self.stopper.step(self.valid_loss_history[self.epoch]):
                    break  # early stop criterion is met, we can stop now

            # logging metrics
            print('epoch [{}/{}]   -- Train: '
                  '{:.8f}, Valid: {:.8f} -- '
                  'lr={:.8f}'.format(self.epoch, self.num_epochs,
                                     self.train_loss_history[self.epoch],
                                     self.valid_loss_history[self.epoch],
                                     self.lr_history[self.epoch]))
            if isinstance(self.lossfunc, list):
                for iloss, lossname in enumerate(self.losses_names):
                    weight = self.lossweights[iloss].item()
                    weight1 = 0. if not self.learn_lossweights else \
                                 np.exp(-self.lossweights[iloss].item()) / 2
                    print('{:2s} (w={:.3f}/{:.3f})  -- Train: {:.8f}/{:.8f}, '
                          'Valid: {:.8f}/{:.8f} --'.format(lossname, weight, weight1,
                                                    self.train_losses_history[lossname][self.epoch],
                                                    self.train_losses_history[lossname][self.epoch] * weight1,
                                                    self.valid_losses_history[lossname][self.epoch],
                                                    self.valid_losses_history[lossname][self.epoch] * weight1))
        self.num_epochs_effective = self.epoch

        def double_train(self, lr_min, lr_max):
            """Train model repeated times (to be used with OneCycle scheduler
            TO BE DONE!
            """
            # train with originally defined learning rates
            self.train()
            self.train_loss_history_tot = self.train_loss_history
            self.valid_loss_history_tot = self.train_loss_history

            # change learning rates
            self.optimizer.lr = lr_max
            self.scheduler.lr_max = lr_max
            print(self.optimizer, self.scheduler)

            # carry on training
            self.train()

            self.train_loss_history_tot = self.train_loss_history_tot + self.train_loss_history
            self.valid_loss_history_tot = self.valid_loss_history_tot + self.valid_loss_history