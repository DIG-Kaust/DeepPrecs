import numpy as np


def subsampling(n, kind='irreg', perc=0.4):
    """Subsampling mask

    Design subsampling mask

    Parameters
    ----------
    n : :obj:`int`
        Total elements to subsample from
    kind : :obj:`str`, optional
        Kind of subsampling (``reg`` and ``irreg``)
    perc : :obj:`float`, optional
        Percentage of subsampling. For regular it must be the denominator (eg for 25% pass 4).

    Returns
    -------
    iava : :obj:`list`
        List of available indices
    mask : :obj:`numpy.ndarray`
        1D mask

    """
    if kind == 'irreg':
        # Irregular restriction operator
        np.random.seed(0)  # fix seed for iava
        nsub = int(np.round(n * perc))
        iava = np.sort(np.random.permutation(np.arange(n))[:nsub])
    elif kind == 'reg':
        # Regular restriction operator
        iava = np.arange(0, n, int(perc))
    else:
        raise NotImplementedError('kind must be reg or irreg')

    mask = np.zeros(n)
    mask[iava] = 1

    return iava, mask