import numpy as np
import torch

from scipy.optimize import minimize, Bounds


class TorchInvert(torch.nn.Module):
    """Inversion torch module

    This class simply wraps the forward problem and defines
    the parameters to invert for

    Parameters
    ----------
    nx : :obj:`int`
        Number of samples of model vector
    op : :obj:`torch.nn.Module`
        Forward operator
    device : :obj:`str`
        Device
    x0 : :obj:`torch.tensor`
        Starting guess (if ``None``, selected randomly)

    """
    def __init__(self, nx, op, device, x0=None):
        super().__init__()
        if x0 is None:
            self.x = torch.nn.Parameter(torch.rand((1, nx), device=device)-0.5)
        else:
            self.x = torch.nn.Parameter(x0)
        self.op = op
        self.neval = 0

    def forward(self):
        self.neval += 1
        y, xdec = self.op(self.x)
        return y, xdec

    def grad(self):
        return self.x.grad


class InvertAll():
    """Invert class

    Solve a Deep Preconditioned inverse problem with either scipy and torch optimizers

    Parameters
    ----------
    device : :obj:`str`
        Device
    nenc : :obj:`int`
        Size of latent code
    npatches : :obj:`int`
        Number of patches in model vector
    model : :obj:`torch.nn.Module`
        Network
    model_forward : :obj:`callable`
        Network forward operator (including physical operator)
    model_rec : :obj:`callable`
        Network forward operator (excluding physical operator)
    lossfunc : :obj:`torch.nn.modules.loss`
        Loss function
    learning_rate : :obj:`float`
        Learning rate (only when using Adam optimizer)
    niter : :obj:`int`, optional
        Number of iterations
    reg_ae : :obj:`float`, optional
        Contibution of regularization term that check consistency of solution
        with the manifold of the autoencoder
    x0 : :obj:`np.ndarray` or :obj:`torch.tensor`, optional
        Starting guess (if ``None``, selected randomly)
    bounds : :obj:`tuple`, optional
        Bounds ``(min, max)`` to apply to scipy optimizer
    model_eval : :obj:`bool`, optional
        Whether to run eval on model or not

    """
    def __init__(self, device,
                 nenc, npatches,
                 model, model_forward, model_rec,
                 lossfunc, learning_rate, niter,
                 reg_ae=0.0,
                 x0=None,
                 bounds=None,
                 model_eval=True,
                 ):
        self.device = device
        self.nenc = nenc
        self.npatches = npatches
        self.model = model
        self.model_forward = model_forward
        self.model_rec = model_rec
        self.lossfunc = lossfunc
        self.learning_rate = learning_rate
        self.niter = niter
        self.reg_ae = reg_ae
        self.x0 = x0
        self.bounds = bounds
        if model_eval:
            self.model.eval()

    @staticmethod
    def fun(x, yobs, model, Op, distance, device, reg_ae=0.):
        x = torch.from_numpy(x.astype(np.float32).reshape(1, x.shape[0])).to(device)
        y, xdec = Op(x)
        j = distance(y.float(), yobs).cpu().detach().numpy()
        if reg_ae > 0:
            xae = model(xdec)
            j += reg_ae * distance(xdec, xae).cpu().detach().numpy()
        return float(j)

    @staticmethod
    def grad(x, yobs, model, Op, distance, device, reg_ae=0.):
        x = torch.tensor(x.astype(np.float32).reshape(1, x.shape[0]),
                         requires_grad=True, device=device)
        y, xdec = Op(x)
        loss = distance(y.float(), yobs)
        if reg_ae > 0:
            xae = model(xdec)
            loss += reg_ae * distance(xdec, xae)
        model.zero_grad()
        loss.backward(retain_graph=True)
        grad = x.grad.cpu().detach().numpy()
        return grad.astype(np.float64)

    def callback(self, x, y, forward, prec, xtrue, device):
        res = y - forward(
            torch.from_numpy(x.astype(np.float32).reshape(1, x.shape[0])).to(
                device))[0].cpu().detach().numpy()
        err = prec(
            torch.from_numpy(x.astype(np.float32).reshape(self.npatches, self.nenc)).to(
                device))[0].cpu().detach().numpy().ravel() - xtrue
        self.resnorm[self.ifeval] = np.linalg.norm(res)
        self.errnorm[self.ifeval] = np.linalg.norm(err)
        self.ifeval += 1

    def scipy_invert(self, d, mtrue):
        """Scipy inversion

        Solve a Deep Preconditioned inverse problem with scipy L-BFGS-B optimizer

        Parameters
        ----------
        d : :obj:`np.ndarray`
            Data
        mtrue : :obj:`np.ndarray`
            True model (to compute error as function of iterations). If
            not available, provide zero-filled vector

        """
        f = lambda x: self.fun(x, d, self.model, self.model_forward,
                               self.lossfunc, self.device, self.reg_ae)
        g = lambda x: self.grad(x, d, self.model, self.model_forward,
                                self.lossfunc, self.device, self.reg_ae)
        if self.bounds is not None:
            self.bounds = Bounds(self.bounds[0]*np.ones(self.nenc*self.npatches),
                                 self.bounds[1]*np.ones(self.nenc*self.npatches))
        self.ifeval = 0
        self.resnorm = np.zeros(self.niter)
        self.errnorm = np.zeros(self.niter)
        if self.x0 is None:
            self.x0 = np.zeros(self.nenc*self.npatches, dtype=np.float32)
        nl = minimize(f, self.x0, jac=g,
                      bounds=self.bounds,
                      method='L-BFGS-B', tol=0,
                      callback=lambda x: self.callback(x, d.cpu().detach().numpy(),
                                                       self.model_forward,
                                                       self.model_rec,
                                                       mtrue.cpu().detach().numpy().ravel(),
                                                       self.device),
                      options=dict(maxfun=self.niter * 2, maxiter=self.niter, disp=False))
        pnl = nl.x
        mnl = self.model_rec(torch.from_numpy(pnl.astype(np.float32)).to(self.device).reshape(self.npatches, self.nenc))[0].cpu().detach().numpy().squeeze()
        return mnl, pnl

    def torch_adam_invert(self, d, mtrue, verb=True):
        """Torch Adam inversion

        Solve a Deep Preconditioned inverse problem with torch Adam optimizer

        Parameters
        ----------
        d : :obj:`np.ndarray`
            Data
        mtrue : :obj:`np.ndarray`
            True model (to compute error as function of iterations). If
            not available, provide zero-filled vector
        verb : :obj:`bool`, optional
            Verbose iterations

        """
        inv = TorchInvert(self.nenc*self.npatches, self.model_forward, self.device, x0=self.x0)
        optimizer = torch.optim.Adam(inv.parameters(), lr=self.learning_rate)

        if verb:
            print('===========================')
            print('Adam')
            print('===========================')
            print('    Iter:    |     F       ')
            print('---------------------------')

        self.resnorm = np.zeros(self.niter)
        self.errnorm = np.zeros(self.niter)
        for i in range(self.niter):
            inv.zero_grad()
            dhat, xdec = inv()
            loss = self.lossfunc(dhat, d)
            if self.reg_ae > 0:
                xae = self.model(xdec)
                loss += self.reg_ae * self.lossfunc(xdec, xae)
            loss.backward()
            self.resnorm[i] = np.sqrt(d.shape[0] * loss.item())
            with torch.no_grad():
                self.errnorm[i] = np.sqrt(np.prod(len(mtrue.view(-1))) * float(self.lossfunc(self.model_rec(inv.x)[0], mtrue.view(-1)).item()))
            optimizer.step()
            if verb and ((i % 10 == 0) or (i < 10) or (i>self.niter-10)):
                # print log
                print('  %6g  |  %.3e' % (i + 1, self.resnorm[i]))
        return self.model_rec(inv.x)[0].cpu().detach().numpy(), \
               inv.x.cpu().detach().numpy().squeeze()

    def torch_lbfgs_invert(self, d, mtrue, verb=True):
        """Torch Adam inversion

        Solve a Deep Preconditioned inverse problem with torch L-BFGS-B optimizer

        Parameters
        ----------
        d : :obj:`np.ndarray`
            Data
        mtrue : :obj:`np.ndarray`
            True model (to compute error as function of iterations). If
            not available, provide zero-filled vector
        verb : :obj:`bool`, optional
            Verbose iterations

        """
        inv = TorchInvert(self.nenc * self.npatches, self.model_forward,
                          self.device, x0=self.x0)
        optimizer = torch.optim.LBFGS(inv.parameters(), lr=self.learning_rate,
                                      history_size=10, max_iter=5,
                                      line_search_fn="strong_wolfe")
        if verb:
            print('===========================')
            print('LGFGS')
            print('===========================')
            print('    Iter:    |     F       ')
            print('---------------------------')

        def f(d):
            dhat, xdec = inv()
            loss = self.lossfunc(dhat, d)
            if self.reg_ae > 0:
                xae = self.model(xdec)
                loss += self.reg_ae * self.lossfunc(xdec, xae)
            return loss

        self.resnorm = np.zeros(self.niter)
        self.errnorm = np.zeros(self.niter)
        for i in range(self.niter):
            optimizer.zero_grad()
            loss = f(d)
            loss.backward()
            optimizer.step(lambda: f(d))
            with torch.no_grad():
                self.resnorm[i] = np.sqrt(
                    d.shape[0] * self.lossfunc(inv()[0], d).item())
                self.errnorm[i] = np.sqrt(np.prod(len(mtrue.view(-1))) * float(
                    self.lossfunc(self.model_rec(inv.x)[0],
                                  mtrue.view(-1)).item()))
            if verb and ((i % 10 == 0) or (i < 10) or (i > self.niter - 10)):
                # print log
                print('  %6g  |  %.3e' % (i + 1, self.resnorm[i]))
        return self.model_rec(inv.x)[0].cpu().detach().numpy(), \
               inv.x.cpu().detach().numpy().squeeze()