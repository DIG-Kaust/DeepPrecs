#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml
import warnings

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as pl_EarlyStopping
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pylops.basicoperators import Identity, Restriction
from pylops.signalprocessing import Patch2D
from pylops.utils.wavelets import ricker
from pylops_gpu import TorchOperator
from scipy.signal import convolve

from deepprecs.ghost import deghosting
from deepprecs.patching import patching, patch_scalings, patch2d_design
from deepprecs.subsampling import subsampling
from deepprecs.aemodel import AutoencoderBase, AutoencoderRes, AutoencoderMultiRes
from deepprecs.train_pl import *
from deepprecs.invert import InvertAll

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def _str_to_tuple(string, dtype=float):
    strings = string.split(',')
    return (dtype(strings[0]), dtype(strings[1]))

def _check_none(value, dtype=float):
    return None if value == 'None' else dtype(value)


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-c', '--config', type=str, help='Configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    # Experiment number and name
    iexp = setup['exp']['number']
    expname = setup['exp']['name']
    label = setup['exp']['label']

    # Global settings
    devicenum = setup['global']['devicenum']
    seed = setup['global']['seed']
    display = setup['global']['display']
    clip = setup['global']['clip']

    # Patches
    nspatch, ntpatch = setup['patching']['nspatch'], setup['patching']['ntpatch']
    nsjump, ntjump = setup['patching']['nsjump'], setup['patching']['ntjump']

    # AE parameters
    if  setup['autoencoder']['aetype'] == 'AutoencoderBase':
        aetype = AutoencoderBase
    elif setup['autoencoder']['aetype'] == 'AutoencoderRes':
        aetype = AutoencoderRes
    elif setup['autoencoder']['aetype'] == 'AutoencoderMultiRes':
        aetype = AutoencoderMultiRes
    else:
        raise NotImplementedError('AEtype must be AutoencoderBase, AutoencoderRes, or AutoencoderMultiRes')

    nenc = setup['autoencoder']['nenc']
    kernel_size = setup['autoencoder']['kernel_size']
    nfilts = setup['autoencoder']['nfilts']
    nlayers = setup['autoencoder']['nlayers']
    nlevels = setup['autoencoder']['nlevels']
    convbias = setup['autoencoder']['convbias']
    downstride = setup['autoencoder']['downstride']
    downmode = setup['autoencoder']['downmode']
    upmode = setup['autoencoder']['upmode']
    bnormlast = setup['autoencoder']['bnormlast']
    act_fun = setup['autoencoder']['act_fun']
    relu_enc = setup['autoencoder']['relu_enc']
    tanh_enc = setup['autoencoder']['tanh_enc']
    relu_dec = setup['autoencoder']['relu_dec']
    tanh_final = setup['autoencoder']['tanh_final']

    # Loss/optimizer
    loss = setup['optimizer']['loss']
    lossweights = _check_none(setup['optimizer']['lossweights'])
    betas = _str_to_tuple(setup['optimizer']['betas'], float)
    weight_decay = float(setup['optimizer']['weight_decay'])
    learning_rate = float(setup['optimizer']['learning_rate'])
    adapt_learning = setup['optimizer']['adapt_learning']
    lr_scheduler = setup['optimizer']['lr_scheduler']
    lr_factor = _check_none(setup['optimizer']['lr_factor'])
    lr_thresh = _check_none(setup['optimizer']['lr_thresh'])
    lr_patience = _check_none(setup['optimizer']['lr_patience'])
    lr_max = _check_none(setup['optimizer']['lr_max'])
    es_patience = setup['optimizer']['es_patience']
    es_min_delta = float(setup['optimizer']['es_min_delta'])

    # Training
    num_epochs = setup['training']['num_epochs']
    batch_size = setup['training']['batch_size']
    noise_std = setup['training']['noise_std']
    mask_perc = setup['training']['mask_perc']

    ####### DEGHOSTING AND INTERPOLATION #######
    # Source index to be used for inversion
    isrc = setup['deghosting']['isrc']
    fwav = setup['deghosting']['fwav']

    # Direct wave window
    toff = setup['deghosting']['toff']
    nsmoothwin = setup['deghosting']['nsmoothwin']

    # Ghost  operator
    vel_sep = setup['deghosting']['vel_sep']
    nxpad = setup['deghosting']['nxpad']
    ntaper = setup['deghosting']['ntaper']

    # Restriction operator
    kind = setup['deghosting']['kind']
    perc = setup['deghosting']['perc']

    # Patching operator
    nwin = (nspatch, ntpatch)
    nover = _str_to_tuple(setup['deghosting']['nover'], int)
    nop = (nspatch, ntpatch)

    # Device
    device = torch.device(f"cuda:{devicenum}" if torch.cuda.is_available() else "cpu")
    print(device)
    if 'cuda' in str(device):
        import cupy as cp

    # Input filepath
    inputfile = setup['exp']['inputfile']
    nr = setup['exp']['nrec']

    # Model and figure directories
    outputmodels = f'../models/{label}'
    outputfigs = f'../figures/{label}'

    # Create directory to save training evolution
    figdir = os.path.join(outputfigs, 'exp%d' % iexp)
    if figdir is not None:
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    # Display experiment setup
    sections = ['exp', 'global', 'patching',
                'autoencoder', 'optimizer',
                'training', 'deghosting']
    print('----------------------------')
    print('DeepPrec Deghosting')
    print('----------------------------\n')
    for section in sections:
        print(section.upper())
        for key, value in setup[section].items():
            print(f'{key} = {value}')
        print('\n----------------------------\n')
    print(f'GPU used: {torch.cuda.get_device_name(device)}')
    print('----------------------------\n')

    # Seed
    seed_everything(seed)

    ######### DATA LOADING #########
    inputdata = np.load(inputfile)

    # Receivers
    r = inputdata['r'][:, :nr]

    # Sources
    s = inputdata['s'][:, :nr]
    ns = s.shape[1]

    # Model
    vel = inputdata['vel']

    # Axes
    t = inputdata['t']
    x, z = inputdata['x'], inputdata['z']
    nt, dt = len(t), t[1] - t[0]
    dx, dz = x[1] - x[0], z[1] - z[0]

    # Load data
    p = inputdata['p'][:nr, :, :nr].transpose(0, 2, 1)
    p /= p.max()

    # Convolve with wavelet
    wav, _, wav_c = ricker(t[:201], fwav)
    p = np.apply_along_axis(convolve, -1, p, wav, mode='full')
    p = p[:, :, wav_c:][:, :, :nt]

    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, gridspec_kw={'width_ratios': [5, 1]})
    axins1 = inset_axes(axs[0],
                        width="70%",  # width = 50% of parent_bbox width
                        height="5%",  # height : 5%
                        bbox_to_anchor=(0.125, -0.63, .6, .5),
                        bbox_transform=axs[0].transAxes)
    im = axs[0].imshow(vel, extent=(x[0], x[-1], z[-1], z[0]), vmin=1000, vmax=3000, cmap='twilight_r')
    axs[0].scatter(r[0, ::5], r[1, ::5], marker='v', s=150, c='b', edgecolors='k')
    axs[0].scatter(s[0, ::5], s[1, ::5], marker='*', s=250, c='r', edgecolors='k')
    axs[0].axis('tight')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('z [m]')
    axs[0].set_title('Model and Geometry')
    axs[0].set_xlim(x[0], x[-1])
    axs[0].set_ylim(z[-1], z[0])
    fig.colorbar(im, cax=axins1, orientation='horizontal', extend="both", fraction=0.046, pad=0.12)
    axs[1].plot(vel[:, len(x) // 2], z, 'k', lw=2)
    axs[1].set_title('Velocity profile')
    axs[1].set_xlabel('V [m/s]')
    plt.savefig(os.path.join(outputfigs, f'{label}_model.eps'), dpi=100, bbox_inches='tight')

    ######### SETTING DEGHOSTING PROBLEM #########

    # Ghost operator
    win, Dupop = deghosting(s, r, isrc, nt, dt, dz, vel_sep, toff, nsmoothwin, nxpad, ntaper)

    # Visualize
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 8))
    axs[0].imshow(p[isrc].T, cmap='gray',
                  vmin=-clip * np.abs(p).max(), vmax=clip * np.abs(p).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[0].set_title(r'$P$')
    axs[0].axis('tight')
    axs[1].imshow(win * p[isrc].T, cmap='gray',
                  vmin=-clip * np.abs(p).max(), vmax=clip * np.abs(p).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[1].set_title(r'Windowed $P$')
    axs[1].axis('tight')
    axs[2].imshow(win, cmap='seismic', vmin=-1, vmax=1,
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[2].set_title(r'Window')
    axs[2].axis('tight')
    axs[2].set_ylim(0.8*t[-1], 0)
    plt.savefig(os.path.join(outputfigs, f'{label}_window.eps'), dpi=100, bbox_inches='tight')

    # Restriction operator
    if kind == 'reg':
        perc = int(1. / perc)
    iava, mask = subsampling(ns, kind=kind, perc=perc)
    Rop = Restriction(ns * nt, iava, dims=(ns, nt), dir=0, dtype='float32')

    plt.figure(figsize=(15, 2))
    plt.plot(mask, 'k')
    plt.title('Selected receivers')
    plt.savefig(os.path.join(outputfigs, f'{label}_selrecs.eps'), dpi=100, bbox_inches='tight')

    # Create full and masked data
    dfull = win.T * p[isrc]
    d = Rop * dfull.ravel()

    dmask = Rop.H * d
    dmask = dmask.reshape(nr, nt)

    # Convert data to torch
    d_torch = torch.from_numpy(d.astype(np.float32)).to(device)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 8))
    axs[0].imshow(dfull.T, cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[0].set_title(r'$d_{full}$')
    axs[0].set_xlabel(r'$x_R$')
    axs[0].set_ylabel(r'$t$')
    axs[0].axis('tight')
    axs[1].imshow(dmask.T, cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[1].set_title(r'$d_{sub}$')
    axs[1].set_xlabel(r'$x_R$')
    axs[1].axis('tight')
    axs[1].set_ylim(0.8*t[-1], 0)
    plt.savefig(os.path.join(figdir, 'exp%d_data.png' % iexp))

    # Patch operator
    dimsd = (ns, nt)
    npatches = patch2d_design(dimsd, (nspatch, ntpatch), nover, nop)[0]
    dims = (npatches[0] * nspatch, npatches[1] * ntpatch)

    Op = Identity(nspatch * ntpatch, dtype='float32')
    Pop = Patch2D(Op, dims, dimsd, nwin, nover, nop,
                  tapertype=None, design=True)
    Pop1 = Patch2D(Op.H, dims, dimsd, nwin, nover, nop,
                   tapertype='cosine', design=False)
    Pop1_torch = TorchOperator(Pop1, pylops=True, device=device)

    # Create patches and revert back
    dfull_patches = Pop.H * dfull.ravel()
    dfull_patches = dfull_patches.reshape(dims)
    dfull_repatched = Pop1 * dfull_patches.ravel()
    dfull_repatched = dfull_repatched.reshape(dimsd)

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 8))
    axs[0].imshow(dfull.T, cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[0].set_title(r'$d_{full}$')
    axs[0].set_xlabel(r'$x_R$')
    axs[0].set_ylabel(r'$t$')
    axs[0].axis('tight')
    axs[1].imshow(dfull_repatched.T, cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[1].set_title(r'$d_{repatched}$')
    axs[1].set_xlabel(r'$x_R$')
    axs[1].axis('tight')
    axs[1].set_ylim(0.8*t[-1], 0)
    axs[2].imshow(dfull.T - dfull_repatched.T, cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[2].set_title(r'$d_{full}-d_{repatched}$')
    axs[2].set_xlabel(r'$x_R$')
    axs[2].axis('tight')
    plt.savefig(os.path.join(figdir, 'exp%d_datapatching.png' % iexp))

    # Find scalings
    scalings = patch_scalings(dmask, Pop, npatches, npatch=(nspatch, ntpatch),
                              plotflag=True, clip=clip, device=device)

    # Overall operator
    RDupop = Rop * Dupop
    RDupop_torch = TorchOperator(RDupop, pylops=True, device=device)

    ######### TRAINING DATASET #########

    # Visualize receiver gathers
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 9))
    axs[0].imshow(p[:, 20].T, cmap='gray', vmin=-clip*np.abs(p).max(), vmax=clip*np.abs(p).max())
    axs[0].set_title('R rec=0'), axs[0].set_xlabel(r'$x_R$'), axs[0].set_ylabel(r'$t$')
    axs[0].axis('tight')
    axs[0].set_xticks(np.arange(0, nr, nspatch))
    axs[0].set_yticks(np.arange(0, nt, ntpatch))
    axs[0].grid(which='both')
    axs[1].imshow(p[:, nr//2].T, cmap='gray', vmin=-clip*np.abs(p).max(), vmax=clip*np.abs(p).max())
    axs[1].set_title('R rec=%d' %(nr//2)), axs[1].set_xlabel(r'$x_R$')
    axs[1].axis('tight')
    axs[1].set_xticks(np.arange(0, nr, nspatch))
    axs[1].set_yticks(np.arange(0, nt, ntpatch))
    axs[1].grid(which='both')
    axs[2].imshow(p[:, -20].T, cmap='gray', vmin=-clip*np.abs(p).max(), vmax=clip*np.abs(p).max())
    axs[2].set_title('R rec=%d' %(nr-20)), axs[2].set_xlabel(r'$x_R$')
    axs[2].axis('tight')
    axs[2].set_xticks(np.arange(0, nr, nspatch))
    axs[2].set_yticks(np.arange(0, nt, ntpatch))
    axs[2].grid(which='both')
    plt.savefig(os.path.join(figdir, 'exp%d_recgathers.png' % iexp))

    # Create patches for training
    xs = patching(p[:, iava], s, r, dt, npatch=(nspatch, ntpatch), njump=(nsjump, ntjump), window=True,
                  vel_sep=vel_sep, toff=toff, nsmoothwin=nsmoothwin, thresh=1e-4, augumentdirect=True)

    nviz = 4
    np.random.seed(5)
    ivizs = np.random.permutation(np.arange(len(xs)))[:nviz ** 2]
    fig, axs = plt.subplots(nviz, nviz, sharey=True, figsize=(8, 8))
    axs = axs.ravel()
    fig.suptitle('Dataset', y=0.99)
    for i, iviz in enumerate(ivizs):
        axs[i].imshow(xs[iviz].T, cmap='gray', vmin=-clip, vmax=clip)
        axs[i].axis('tight')
    plt.savefig(os.path.join(figdir, 'exp%d_scalings.png' % iexp))

    ######### TRAINING #########

    # Create directory to save training evolution
    figdir = os.path.join(outputfigs, 'exp%d' % iexp)

    if figdir is not None:
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    # Create model to train
    autoencoder = aetype(nh=ntpatch, nw=nspatch, nenc=nenc,
                         kernel_size=kernel_size, nfilts=nfilts,
                         nlayers=nlayers, nlevels=nlevels,
                         physics=RDupop_torch,
                         convbias=convbias, act_fun=act_fun,
                         downstride=downstride, downmode=downmode,
                         upmode=upmode, bnormlast=bnormlast,
                         relu_enc=relu_enc, tanh_enc=tanh_enc,
                         relu_dec=relu_dec, tanh_final=tanh_final,
                         patcher=Pop1_torch, npatches=npatches[0] * npatches[1],
                         patchesscaling=scalings)

    # Create dataset
    datamodule = DataModule(xs, valid_size=0.1, random_state=42, batch_size=batch_size)

    # Callbacks
    early_stop_callback = pl_EarlyStopping(monitor="val_loss",
                                           mode="min",
                                           patience=es_patience,
                                           min_delta=es_min_delta)
    callback = MetricsCallback(loss)
    callback1 = PlottingCallback(figdir, notebook=False)

    # Training
    dimred = LitAutoencoder(ntpatch, nspatch, nenc,
                            autoencoder, loss, num_epochs, lossweights=lossweights,
                            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas,
                            adapt_learning=True, lr_scheduler=lr_scheduler, lr_factor=lr_factor,
                            lr_thresh=lr_thresh, lr_patience=lr_patience, lr_max=lr_max,
                            noise_std=noise_std, mask_perc=mask_perc, device=device)

    trainer = pl.Trainer(accelerator='gpu', devices=[devicenum, ],
                         max_epochs=dimred.num_epochs, log_every_n_steps=4,
                         callbacks=[early_stop_callback, callback, callback1])
    trainer.fit(dimred, datamodule)

    # Save model
    torch.save(autoencoder.state_dict(), os.path.join(outputmodels, 'exp%d_modelweights.pt' % iexp))

    # Predict by manually piping encoder and decoder and visualize
    nstats = 100
    nviz = 5
    autoencoder.cuda(devicenum)
    autoencoder.eval()
    x_train = datamodule.x_train
    x_valid = datamodule.x_valid

    x_train_latent = torch.from_numpy(x_train[:nstats].astype(np.float32)).view(nstats, 1, nspatch, ntpatch).to(
        device)
    latent = autoencoder.encode(x_train_latent)
    x_train_ae = autoencoder.decode(latent)
    x_train_mse = np.linalg.norm(
        x_train[:nstats].ravel() - x_train_ae[:nstats].detach().cpu().numpy().ravel()) / nstats

    fig, axs = plt.subplots(3, nviz, sharey=True, figsize=(16, 8))
    plt.suptitle(f'Training Reconstruction exp{iexp} (MSE={x_train_mse:.4f})',
                 y=0.95, fontsize=15, fontweight='bold')
    for i in range(nviz):
        # display original
        axs[0, i].imshow(x_train[i].reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[0, i].axis('tight')
        # display reconstruction
        axs[1, i].imshow(x_train_ae[i].cpu().detach().numpy().reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[1, i].axis('tight')
        # display error
        axs[2, i].imshow(x_train[i].reshape(nspatch, ntpatch).T -
                         x_train_ae[i].cpu().detach().numpy().reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[2, i].axis('tight')
    plt.savefig(os.path.join(figdir, 'exp%d_train.png' % iexp))

    x_valid_latent = torch.from_numpy(x_valid[:nstats].astype(np.float32)).view(nstats, 1, nspatch, ntpatch).to(
        device)
    latent1 = autoencoder.encode(x_valid_latent)
    x_valid_ae = autoencoder.decode(latent1)
    x_valid_mse = np.linalg.norm(
        x_valid[:nstats].ravel() - x_valid_ae[:nstats].detach().cpu().numpy().ravel()) / nstats

    fig, axs = plt.subplots(3, nviz, sharey=True, figsize=(16, 8))
    plt.suptitle(f'Validation Reconstruction exp{iexp} (MSE={x_valid_mse:.4f})',
                 y=0.95, fontsize=15, fontweight='bold')
    for i in range(nviz):
        # display original
        axs[0, i].imshow(x_valid[i].reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[0, i].axis('tight')
        # display reconstruction
        axs[1, i].imshow(x_valid_ae[i].cpu().detach().numpy().reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[1, i].axis('tight')
        # display error
        axs[2, i].imshow(x_valid[i].reshape(nspatch, ntpatch).T -
                         x_valid_ae[i].cpu().detach().numpy().reshape(nspatch, ntpatch).T, cmap='gray',
                         vmin=-clip, vmax=clip)
        axs[2, i].axis('tight')
    plt.savefig(os.path.join(figdir, 'exp%d_valid.png' % iexp))

    # Display training losses
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(callback.train_loss, 'k', lw=2, ms=10, label='Train')
    ax.plot(callback.valid_loss, 'r', lw=2, ms=10, label='Valid')
    ax.set_title('Losses')
    ax.set_xlabel('Epochs')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'exp%d_trainingvalidhistory.png' % iexp))

    if isinstance(dimred.lossfunc, list):
        fig, ax = plt.subplots(1, 1 if lossweights is None else 2, figsize=(16, 5))
        ls = ['-', '-.', '--']
        for iloss, lossname in enumerate(dimred.losses_names):
            ax.plot(callback.train_losses[lossname], 'k', lw=2, ms=10,
                    linestyle=ls[iloss], label='Train %s' % lossname)
            ax.plot(callback.valid_losses[lossname], 'r', lw=2, ms=10,
                    linestyle=ls[iloss], label='Valid %s' % lossname)
            ax.set_title('Losses')
        ax.legend()
        ax.set_xlabel('Epochs')
        plt.savefig(os.path.join(figdir, 'exp%d_losses.png' % iexp))


    ######### DEGHOSTING #########
    print('Deghosting...')
    autoencoder.train()
    autoencoder.eval()

    # Initial guess
    patchesmask = Pop.H * dmask.ravel()
    patchesmask = patchesmask.reshape(npatches[0]*npatches[1], 1, nspatch, ntpatch)
    patchesmask_scaled = patchesmask / scalings.cpu().detach().numpy()
    p0 = autoencoder.encode(torch.from_numpy(patchesmask_scaled.astype(np.float32)).to(device)).cpu().detach().numpy()

    # Invert
    inv = InvertAll(device, # device
                    nenc, npatches[0] * npatches[1],
                    autoencoder, autoencoder.patched_physics_decode, autoencoder.patched_decode, # modelling ops
                    nn.MSELoss(), 1., 80, # optimizer
                    reg_ae=0., x0=p0, bounds=None
                    )
    minv, pinv = inv.scipy_invert(d_torch, torch.zeros(((nr, nt))).to(device))

    # Recompute data from minv
    if 'cuda' in str(device):
        dinv = cp.asnumpy(Dupop * cp.asarray(minv))
    else:
        dinv = Dupop * minv

    minv = minv.reshape(ns, nt)
    dinv = dinv.reshape(ns, nt)

    # Visualize
    pad = 4
    tgain = np.exp(0.5 * t)

    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(15, 9))
    axs[0].imshow(dfull[pad:-pad].T * tgain[:, np.newaxis], cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[0].set_title('d ghost')
    axs[0].set_xlabel(r'$x_R$')
    axs[0].set_ylabel(r'$t$')
    axs[0].axis('tight')
    axs[1].imshow(dmask[pad:-pad].T * tgain[:, np.newaxis], cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[1].set_title('d ghost masked')
    axs[1].set_xlabel(r'$x_R$')
    axs[1].axis('tight')
    axs[2].imshow(minv[pad:-pad].T * tgain[:, np.newaxis], cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[2].set_title('minv')
    axs[2].set_xlabel(r'$x_R$')
    axs[2].axis('tight')
    axs[3].imshow(dinv[pad:-pad].T * tgain[:, np.newaxis], cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[3].set_title('minv reghosted')
    axs[3].set_xlabel(r'$x_R$')
    axs[3].axis('tight')
    axs[4].imshow((dfull[pad:-pad].T - dinv[pad:-pad].T) * tgain[:, np.newaxis], cmap='gray',
                  vmin=-clip * np.abs(dfull).max(), vmax=clip * np.abs(dfull).max(),
                  extent=(r[0, 0], r[0, -1], t[-1], t[0]))
    axs[4].set_title('res')
    axs[4].set_xlabel(r'$x_R$')
    axs[4].axis('tight')
    axs[4].set_ylim(0.8*t[-1], 0)
    plt.savefig(os.path.join(figdir, f'exp{iexp}_deghosting.png'))

    plt.figure()
    plt.plot(inv.resnorm, 'k')
    plt.title(f'Resnorm exp{iexp}')
    plt.savefig(os.path.join(figdir, f'exp{iexp}_deghostingresnorm.png'))

    if display:
        plt.show()


if __name__ == "__main__":
    description = 'DeepPrec Deghosting'
    main(argparse.ArgumentParser(description=description))

