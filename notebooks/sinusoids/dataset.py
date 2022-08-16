import numpy as np
import pylops


def prior_realization(f0, a0, phi0, sigmaf, sigmaa, sigmaphi, dt, nt, nfft):
    """Draw superposition of sinusoids in frequency domain

    Draw a realization given prior knowledge of the family of sinusoids parametrized
    via their frequency, amplitude and phase

    Parameters
    ----------
    f0 : :obj:`list`
        List of averages of the frequencies for each of the sinusoids to sum
    a0 : :obj:`list`
        List of averages of the amplitudes for each of the sinusoids to sum
    phi0 : :obj:`list`
        List of averages of the phases for each of the sinusoids to sum
    sigmaf : :obj:`list`
        List of standard deviations of the frequencies for each of the sinusoids to sum
    sigmaa : :obj:`list`
        List of standard deviations of the amplitudes for each of the sinusoids to sum
    sigmaphi : :obj:`list`
        List of standard deviations of the phases for each of the sinusoids to sum
    sigmaphi : :obj:`float`
        Time sampling
    nt : :obj:`float`
        Number of time samples
    nfft : :obj:`float`
        Number of frequency samples

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Realization in time domain

    """
    f = np.fft.rfftfreq(nfft, dt)
    df = f[1] - f[0]

    # draw frequencies, amplitudes and phases of the sinusoids
    ifreqs = [int(np.random.normal(f, sigma) / df) for f, sigma in zip(f0, sigmaf)]
    amps = [np.random.normal(a, sigma) for a, sigma in zip(a0, sigmaa)]
    phis = [np.random.normal(phi, sigma) for phi, sigma in zip(phi0, sigmaphi)]

    # create input signal in frequency domain
    X = np.zeros(nfft // 2 + 1, dtype='complex128')
    X[ifreqs] = np.array(amps).squeeze() * np.exp(1j * np.deg2rad(np.array(phis))).squeeze()

    # convert input signal to time domain
    FFTop = pylops.signalprocessing.FFT(nt, nfft=nfft, real=True, engine='numpy')
    x = FFTop.H * X
    return x