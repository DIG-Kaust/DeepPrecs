import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import BlockDiag, HStack, Pad
from pylops.signalprocessing import Shift
from pylops.utils.backend import get_array_module


class BlendingContinuous(LinearOperator):
    r"""Continuous blending operator

    Blend seismic shot gathers in continuous mode based on pre-defined sequence of firing times.
    The size of input model vector must be :math:`n_s \times n_r \times n_t`, whilst the size of the data
    vector is :math:`n_r \times n_{t,tot}`.

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples
    nr : :obj:`int`
        Number of receivers
    ns : :obj:`int`
        Number of sources
    dt : :obj:`float`
        Time sampling in seconds
    times : :obj:`np.ndarray`
        Absolute ignition times for each source
    nproc : :obj:`int`, optional
        Number of processors used when applying operator
    dtype : :obj:`str`, optional
        Operator dtype
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Notes
    -----
    Simultaneous shooting or blending is the process of acquiring seismic data firing consecutive sources
    at short time intervals (shorter than the time requires for all significant waves to come back from the Earth
    interior).

    Continuous blending refers to an acquisition scenario where a source towed behind a single vessel is fired at
    irregular time intervals (``times``) to create a continuous recording whose modelling operator is

      .. math::
        \Phi = [\Phi_1, \Phi_2, ..., \Phi_N]

    where each :math:`\Phi_i` operator applies a time-shift equal to the absolute ignition time provided in the
    variable ``times``.

    """

    def __init__(
        self,
        nt,
        nr,
        ns,
        dt,
        times,
        dtype="float64",
        name="B",
    ):
        self.dtype = np.dtype(dtype)
        self.nt = nt
        self.nr = nr
        self.ns = ns
        self.dt = dt
        self.times = times
        self.nttot = int(np.max(self.times) / self.dt + self.nt + 1)
        self.PadOp = Pad((self.nr, self.nt), ((0, 0), (0, 1)), dtype=self.dtype)
        # Define shift operators
        self.shifts = []
        self.ShiftOps = []
        for i in range(self.ns):
            shift = self.times[i]
            # This is the part that fits on the grid
            shift_int = int(shift // self.dt)
            self.shifts.append(shift_int)
            # This is the fractional part
            diff = (shift / self.dt - shift_int) * self.dt
            if diff == 0:
                self.ShiftOps.append(None)
            else:
                self.ShiftOps.append(
                    Shift(
                        (self.nr, self.nt + 1),
                        diff,
                        dir=1,
                        sampling=self.dt,
                        real=False,
                        dtype=self.dtype,
                    )
                )
        self.dtype = np.dtype(dtype)
        self.shape = (self.nr * self.nttot, self.ns * self.nr * self.nt)
        self.name = name

    def _matvec(self, x):
        ncp = get_array_module(x)
        x = x.reshape(self.ns, self.nr, self.nt)
        blended_data = ncp.zeros((self.nr, self.nttot), dtype=self.dtype)
        for i, shift_int in enumerate(self.shifts):
            if self.ShiftOps[i] is None:
                blended_data[:, shift_int : shift_int + self.nt] += x[i, :, :]
            else:
                shifted_data = (self.ShiftOps[i] * self.PadOp * x[i, :, :].ravel()).reshape(self.nr, self.nt + 1)
                blended_data[:, shift_int : shift_int + self.nt + 1] += shifted_data
        return blended_data.real.ravel()

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        x = x.reshape(self.nr, self.nttot)
        deblended_data = ncp.zeros((self.ns, self.nr, self.nt), dtype=self.dtype)
        for i, shift_int in enumerate(self.shifts):
            if self.ShiftOps[i] is None:
                deblended_data[i, :, :] = x[:, shift_int : shift_int + self.nt]
            else:
                shifted_data = (
                    self.PadOp.H
                    * self.ShiftOps[i].H
                    * x[:, shift_int : shift_int + self.nt + 1].ravel()
                ).reshape(self.nr, self.nt)
                deblended_data[i, :, :] = shifted_data
        return deblended_data.real.ravel()


def BlendingHalf(
    nt,
    nr,
    ns,
    dt,
    times,
    group_size,
    n_groups,
    nproc=1,
    dtype="float64",
    name="B",
) -> LinearOperator:
    r"""Half blending operator

    Blend seismic shot gathers in half blending mode based on pre-defined
    sequence of firing times. This type of blending assumes that there are
    multiple sources at different spatial locations firing at the same time.
    This means that the blended data only partially overlaps in space.
    The size of input model vector must be :math:`n_s \times n_r \times n_t`,
    whilst the size of the data vector is :math:`n_{groups} \times n_r \times n_{t,tot}`.

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples
    nr : :obj:`int`
        Number of receivers
    ns : :obj:`int`
        Number of sources. Equal to group_size x n_groups
    dt : :obj:`float`
        Time sampling in seconds
    times : :obj:`np.ndarray`
        Absolute ignition times for each source. This should have dimensions
        :math`n_{groups} \times group_{size}`, where each row contains the firing
        times for every group.
    group_size : :obj:`int`
        The number of sources per group
    n_groups : :obj:`int`
        The number of groups
    nproc : :obj:`int`, optional
        Number of processors used when applying operator
    dtype : :obj:`str`, optional
        Operator dtype
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    Bop : :obj:`pylops.LinearOperator`
        Blending operator

    Notes
    -----
    Simultaneous shooting or blending is the process of acquiring seismic data firing consecutive sources
    at short time intervals (shorter than the time requires for all significant waves to come back from the Earth
    interior).

    Half blending refers to an acquisition scenario where two or more vessels, each with a source are fired at
    short time differences. The same experiment is repeated :math:`n_{groups}` times to create :math:`n_{groups}`
    blended recordings. For the case of 2 sources and an overall number of :math:`N=n_{groups}*group_{size}` shots

    .. math::
        \Phi = \begin{bmatrix}
        \Phi_1     & \mathbf{0}   & \mathbf{0}    & ...          & \Phi_{N/2}  & \mathbf{0}   & \mathbf{0}    &  \\
        \mathbf{0} & \Phi_2       & \mathbf{0}    &              & \mathbf{0}  & \Phi_{N/2+1} & \mathbf{0}  \\
        ...        & ...          & ...           & ...          & ...         & ...          & ...  \\
        \mathbf{0} & \mathbf{0}   & \mathbf{0}    & \Phi_{N/2-1} & \mathbf{0}  & \mathbf{0}   & \Phi_{N} \\
        \end{bmatrix}

    where each :math:`\Phi_i` operator applies a time-shift equal to the absolute ignition time provided in the
    variable ``times``.

    """
    if times.shape[0] != group_size:
        raise ValueError("The first dimension of times must equal group_size")

    Bop = []
    for j in range(group_size):
        OpShift = []
        for i in range(n_groups):
            ShiftOp = Shift(
                (nr, nt), times[j, i], dir=1, sampling=dt, real=False, dtype=dtype
            )
            OpShift.append(ShiftOp)
        Dop = BlockDiag(OpShift, nproc=nproc)
        Bop.append(Dop)
    Bop = HStack(Bop)
    Bop.shape = (n_groups * nr * nt, ns * nr * nt)
    Bop.name = name
    return Bop