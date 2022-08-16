import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.signal import filtfilt


def patching(p, s, r, dt, npatch=(64, 64), njump=(16, 16), window=True,
             vel_sep=1500, toff=0.06, nsmoothwin=5, thresh=1e-4, augumentdirect=True):
    """Create patches from seismic data
    """
    ns, nr, nt = p.shape
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
            pwin = win_.T * p[:, irec, :]
        else:
            pwin = p[:, irec, :]

        # Create patches
        for isrc in range(0, ns - nspatch, nsjump):
            for it in range(0, nt - ntpatch, ntjump):
                # extract patch
                patch = pwin[isrc:isrc + nspatch, it:it + ntpatch]
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
                patch = p[isrc - nspatch // 2:isrc + nspatch // 2,
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