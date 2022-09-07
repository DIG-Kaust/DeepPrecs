import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.signal import filtfilt


def patching(data, s, r, dt, npatch=(64, 64), njump=(16, 16), window=True,
             vel_sep=1500, toff=0.06, nsmoothwin=5, thresh=1e-4, augumentdirect=True):
    """Create patches from seismic data

    Create a set of patches from a seismic dataset to be used as training data

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size :math:`n_s \times n_r \times n_t`
    s : :obj:`numpy.ndarray`
        Sources of size :math:`2 \times n_s`
    r : :obj:`numpy.ndarray`
        Receivers of size :math:`2 \times n_s`
    dt : :obj:`float`
        Time sampling
    npatch : :obj:`tuple`, optional
        Patch size
    njump : :obj:`tuple`, optional
        Jump between patches
    window : :obj:`bool`, optional
        Apply window to remove direct arrival
    vel_sep : :obj:`float`, optional
        Velocity at separation level
    toff : :obj:`float`, optional
        Time offset to apply to the direct wave window
    nsmoothwin : :obj:`int`, optional
        Number of samples of the smoothing filter to apply to direct wave window
    thresh : :obj:`float`, optional
        Treshold used to check if patch is empty and therefore discarded
    augumentdirect : :obj:`bool`, optional
        Add more patches around the direct arrival

    """
    ns, nr, nt = data.shape
    nspatch, ntpatch = npatch
    nsjump, ntjump = njump

    # Create patches
    xs = []
    scales = []
    for irec in range(nr):
        if window:
            # Create direct arrival and window data
            direct = np.sqrt(np.sum((s - r[:, irec:irec + 1]) ** 2, axis=0)) / vel_sep
            direct_off = direct + toff
            win_ = np.zeros((nt, ns))
            iwin = np.round(direct_off / dt).astype(np.int)
            for i in range(ns):
                win_[iwin[i]:, i] = 1
            win_ = filtfilt(np.ones(nsmoothwin) / float(nsmoothwin), 1, win_, axis=0)
            win_ = filtfilt(np.ones(nsmoothwin) / float(nsmoothwin), 1, win_, axis=1)
            datawin = win_.T * data[:, irec, :]
        else:
            datawin = data[:, irec, :]

        # Create patches
        for isrc in range(0, ns - nspatch, nsjump):
            for it in range(0, nt - ntpatch, ntjump):
                # extract patch
                patch = datawin[isrc:isrc + nspatch, it:it + ntpatch]
                scale = np.max(np.abs(patch))
                # check if non-empty patch and add to list of patches
                if np.sum(np.abs(patch) < thresh) < 0.9 * nspatch * ntpatch:
                    scales.append(scale)
                    xs.append(patch / scale)
    print('Nsamples: %d' % len(xs))

    # Add more events near direct wave (strongly dipping events add more)
    ## TODO: Should consider adding the same window as above
    if augumentdirect:
        for irec in range(nr):
            for isrc in range(nspatch // 2, ns - nspatch // 2, nsjump):
                direct = np.sqrt(np.sum((s[:, isrc] - r[:, irec]) ** 2)) / vel_sep
                it = int(direct / dt + np.random.uniform(-5, 5))
                if it < ntpatch // 2:
                    it = ntpatch // 2
                if it > nt - ntpatch // 2:
                    it = nt - ntpatch // 2
                patch = data[isrc - nspatch // 2:isrc + nspatch // 2,
                        irec, it - ntpatch // 2:it + ntpatch // 2]
                scale = np.max(np.abs(patch))
                # check if non-empty patch and add
                if np.sum(np.abs(patch) < thresh) < 0.9 * nspatch * ntpatch:
                    scales.append(scale)
                    xs.append(patch / scale)
                    # add horizontally flipped
                    scales.append(scale)
                    xs.append(np.flipud(patch) / scale)
                    # add negated
                    scales.append(scale)
                    xs.append(-patch / scale)
                    scales.append(scale)
                    xs.append(np.flipud(patch) / scale)

    xs = np.array(xs)
    print('Nsamples (after augumentation): %d' % xs.shape[0])

    return xs


def patch_scalings(data, Pop, npatches, npatch=(64, 64), plotflag=False, clip=0.1, device='cpu'):
    """Find patches scalings

    Find scalings for each patch to apply after decoder

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size :math:`n_s \times n_r \times n_t`
    Pop : :obj:`pylops.LinearOperator`
        Patching operator
    npatches : :obj:`tuple`
        Number of patches along both axes
    npatch : :obj:`tuple`, optional
        Patch size
    plotflag : :obj:`tuple`, optional
        Display random patches
    clip : :obj:`clip`, optional
        Clipping used in display
    device : :obj:`str`, optional
        Device

    """
    nspatch, ntpatch = npatch

    # Create patches
    patches = Pop.H * data.ravel()
    patches_inend = np.arange(0, nspatch * ntpatch * npatches[0] * npatches[1], nspatch * ntpatch)
    patches_inend = np.append(patches_inend, nspatch * ntpatch * npatches[0] * npatches[1])

    # Find scalings
    scalings = np.zeros(npatches[0] * npatches[1])

    if plotflag:
        fig, axs = plt.subplots(npatches[0], npatches[1], figsize=(16, 4))
        axs = axs.ravel()
    for ipatch, (patch_in, patch_end) in enumerate(zip(patches_inend[:-1], patches_inend[1:])):
        scalings[ipatch] = np.max(np.abs(patches[patch_in:patch_end]))
        if scalings[ipatch] == 0.: scalings[ipatch] = 1.
        if plotflag:
            axs[ipatch].imshow(patches[patch_in:patch_end].reshape(nspatch, ntpatch).T / scalings[ipatch],
                           cmap='gray', vmin=-clip * np.abs(data).max(), vmax=clip * np.abs(data).max())
            axs[ipatch].axis('tight')
            axs[ipatch].axis('off')
    scalings = torch.from_numpy(scalings.astype(np.float32)).to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    print(f'Scalings {scalings.squeeze()}')

    return scalings


def _slidingsteps(ntr, nwin, nover):
    """Identify sliding window initial and end points given overall
    trace length, window length and overlap
    Parameters
    ----------
    ntr : :obj:`int`
        Number of samples in trace
    nwin : :obj:`int`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of window
    Returns
    -------
    starts : :obj:`np.ndarray`
        Start indices
    ends : :obj:`np.ndarray`
        End indices
    """
    if nwin > ntr:
        raise ValueError(f"nwin={nwin} is bigger than ntr={ntr}...")
    step = nwin - nover
    starts = np.arange(0, ntr - nwin + 1, step, dtype=int)
    ends = starts + nwin
    return starts, ends


def patch2d_design(dimsd, nwin, nover, nop):
    """Design Patch2D operator
    This routine can be used prior to creating the :class:`pylops.signalprocessing.Patch2D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.
    Parameters
    ----------
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data.
    nwin : :obj:`tuple`
        Number of samples of window.
    nover : :obj:`tuple`
        Number of samples of overlapping part of window.
    nop : :obj:`tuple`
        Size of model in the transformed domain.

    Returns
    -------
    nwins : :obj:`tuple`
        Number of windows.
    dims : :obj:`tuple`
        Shape of 2-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches (stored as tuple of tuples).
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches (stored as tuple of tuples).
    """
    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    dwins_inends = ((dwin0_ins, dwin0_ends), (dwin1_ins, dwin1_ends))
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = (nwins0, nwins1)

    # model windows
    dims = (nwins0 * nop[0], nwins1 * nop[1])
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)
    mwins_inends = ((mwin0_ins, mwin0_ends), (mwin1_ins, mwin1_ends))

    # print information about patching
    logging.warning("%d-%d windows required...", nwins0, nwins1)
    logging.warning(
        "data wins - start:%s, end:%s / start:%s, end:%s",
        dwin0_ins,
        dwin0_ends,
        dwin1_ins,
        dwin1_ends,
    )
    logging.warning(
        "model wins - start:%s, end:%s / start:%s, end:%s",
        mwin0_ins,
        mwin0_ends,
        mwin1_ins,
        mwin1_ends,
    )
    return nwins, dims, mwins_inends, dwins_inends



