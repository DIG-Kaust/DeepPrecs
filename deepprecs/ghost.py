from scipy.signal import filtfilt
from pylops.utils.tapers import *
from pylops.basicoperators import *
from pylops.waveeqprocessing.oneway import PhaseShift
from pylops.optimization.sparsity import *


def deghosting(s, r, isrc, nt, dt, dz, vel_sep, toff, nsmoothwin, nxpad, ntaperghost):
    """Ghost operator

    Create ghost operator

    Parameters
    ----------
    s : :obj:`numpy.ndarray`
        Sources of size :math:`2 \times n_s`
    r : :obj:`numpy.ndarray`
        Receivers of size :math:`2 \times n_s`
    isrc : :obj:`int`
        Index of source to use to create ghost operator
    nt : :obj:`int`
        Number of time samples
    dt : :obj:`float`
        Time sampling
    dz : :obj:`float`
        Depth sampling
    vel_sep : :obj:`float`
        Velocity at separation level
    toff : :obj:`float`
        Time offset to apply to the direct wave window
    nsmoothwin : :obj:`int`
        Number of samples of the smoothing filter to apply to direct wave window
    nxpad : :obj:`int`
        Number of samples to use to pad data along receiver axis prior to applying ghost operator
    ntaperghost : :obj:`int`
        Number of samples to use to taper data along receiver axis prior to applying ghost operator

    Returns
    -------
    win : :obj:`numpy.ndarray`
        Direct wave window
    Dupop : :obj:`pylops.LinearOperator`
        Ghost operator
    """
    nr = r.shape[1]
    nf = nt

    # Direct wave window
    direct = np.sqrt(np.sum((s[:, isrc:isrc + 1] - r) ** 2, axis=0)) / vel_sep
    direct_off = direct + toff
    win = np.zeros((nt, nr))
    iwin = np.round(direct_off / dt).astype(np.int)
    for i in range(nr):
        win[iwin[i]:, i] = 1

    win = filtfilt(np.ones(nsmoothwin) / float(nsmoothwin), 1, win, axis=0)
    win = filtfilt(np.ones(nsmoothwin) / float(nsmoothwin), 1, win, axis=1)

    # Ghost model
    zrec = r[1, 0] + dz
    dx = r[0, 1] - r[0, 0]
    nkx = nr + 2 * nxpad
    freq = np.fft.rfftfreq(nf, dt).astype(np.float32)
    kx = np.fft.ifftshift(np.fft.fftfreq(nkx, dx)).astype(np.float32)

    taper = taper2d(nt, nr, ntaperghost).astype(np.float32)
    zprop = 2 * zrec

    Top = Transpose((nr, nt), (1, 0), dtype='float32')
    Padop = Pad((nt, nr), ((0, 0), (nxpad, nxpad)), dtype='float32')
    Pop = - Top.H * Padop.H * PhaseShift(vel_sep, zprop, nt, freq, kx, dtype='float32') * \
          Padop * Diagonal(taper.T.ravel(), dtype='float32') * Top

    Dupop = Identity(nt * nr, dtype='float32') + Pop

    return win, Dupop