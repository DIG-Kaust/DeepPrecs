import numpy as np
import torch

from scipy.optimize import minimize, Bounds
from deepprecs.LBFGS import FullBatchLBFGS


class TorchInvert(torch.nn.Module):
    """Inversion torch module (simply wraps the forward problem and defines
    the parameters to invert for)
    """
    def __init__(self, nx, op, device, x0=None):
        super().__init__()
        if x0 is None:
            #self.x = torch.nn.Parameter(torch.zeros((1, nx), device=device))
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
    """Invert class for entire model
    """
    def __init__(self, device, # device
                 nenc, npatches, # model size
                 model, model_forward, model_rec, # modelling operators
                 lossfunc, learning_rate, niter, # optimizer
                 reg_ae=0.0, # regularization
                 x0=None, # initial guess
                 bounds=None, # bounds (min, max) or None for not using it
                 model_eval=True # whether to run eval on model
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
        self.learning_rate = learning_rate
        self.reg_ae = reg_ae
        self.x0 = x0
        self.bounds = bounds
        if model_eval:
            self.model.eval()

    @staticmethod
    def fun(x, yobs, model, Op, distance, device, reg_ae=0.):
        x = torch.from_numpy(x.astype(np.float32).reshape(1, x.shape[0])).to(device)
        y, xdec = Op(x)
        j = distance(y, yobs).cpu().detach().numpy()
        if reg_ae > 0:
            xae = model(xdec)
            j += reg_ae * distance(xdec, xae).cpu().detach().numpy()
        return float(j)

    @staticmethod
    def grad(x, yobs, model, Op, distance, device, reg_ae=0.):
        x = torch.tensor(x.astype(np.float32).reshape(1, x.shape[0]),
                         requires_grad=True, device=device)
        y, xdec = Op(x)
        loss = distance(y, yobs)
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

    @staticmethod
    def funreg(x, yobs, model, model_rec, Op, distance, device, reg_ae=0.):
        x = torch.from_numpy(x.astype(np.float32)).to(device)
        y = Op(x)
        j = distance(y, yobs).cpu().detach().numpy()
        # add regularization
        j += reg_ae * distance(model_rec(x), x).cpu().detach().numpy()
        return float(j)

    @staticmethod
    def gradreg(x, yobs, model, model_rec, Op, distance, device, reg_ae=0.):
        x = torch.tensor(x.astype(np.float32), requires_grad=True, device=device)
        y = Op(x)
        loss = distance(y, yobs)
        # add regularization
        loss += reg_ae * distance(model_rec(x), x)
        model.zero_grad()
        loss.backward(retain_graph=True)
        grad = x.grad.cpu().detach().numpy()
        return grad.astype(np.float64)

    def callbackreg(self, x, y, forward, xtrue, device):
        res = y - forward(torch.from_numpy(x.astype(np.float32).reshape(x.shape[0]))
                          .to( device))[0].cpu().detach().numpy()
        err = x.ravel() - xtrue
        self.resnorm[self.ifeval] = np.linalg.norm(res)
        self.errnorm[self.ifeval] = np.linalg.norm(err)
        self.ifeval += 1

    def scipy_reginvert(self, d, mtrue):
        f = lambda x: self.funreg(x, d, self.model, self.model_rec, self.model_forward,
                                  self.lossfunc, self.device, self.reg_ae)
        g = lambda x: self.gradreg(x, d, self.model, self.model_rec, self.model_forward,
                                   self.lossfunc, self.device, self.reg_ae)
        self.ifeval = 0
        self.resnorm = np.zeros(self.niter)
        self.errnorm = np.zeros(self.niter)
        if self.x0 is None:
            raise NotImplementedError('Provide x0 explicitely..')
        nl = minimize(f, self.x0, jac=g,
                      bounds=self.bounds,
                      method='L-BFGS-B', tol=0,
                      callback=lambda x: self.callbackreg(x,
                                                          d.cpu().detach().numpy(),
                                                          self.model_forward,
                                                          mtrue.cpu().detach().numpy().ravel(),
                                                          self.device),
                      options=dict(maxfun=self.niter * 2, maxiter=self.niter,
                                   disp=True))
        mnl = nl.x
        return mnl, False

    def torch_adam_invert(self, d, mtrue, verb=True):
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

    def torch_fullbatchlbfgs_invert(self, d, mtrue, line_search='Wolfe', tol=1e-4,
                                    interpolate=True, history_size=10, max_ls=100,
                                    debug=False, verb=True):
        inv = TorchInvert(self.nenc * self.npatches, self.model_forward, self.device, x0=self.x0)
        optimizer = FullBatchLBFGS(inv.parameters(), lr=self.learning_rate,
                                   history_size=history_size,
                                   line_search=line_search, debug=debug)

        # initial cost function and gradient
        optimizer.zero_grad()
        obj = self.lossfunc(inv()[0], d)
        obj.backward()
        grad = inv.grad()
        self.func_evals = 1

        x_old = inv.x.clone()
        x_new = x_old.clone()
        f_old = obj

        if verb:
            print('===================================================================================')
            print('FullBatchLBFGS')
            print('===================================================================================')
            print('    Iter:    |     F       |    ||g||    | |x - xnew|/|x| |   F Evals   |    LR       ')
            print('-----------------------------------------------------------------------------------')

        # main loop
        self.resnorm = np.zeros(self.niter)
        self.errnorm = np.zeros(self.niter)
        for i in range(self.niter):
            # define closure for line search
            def closure():
                optimizer.zero_grad()
                dhat, xdec = inv()
                loss = self.lossfunc(dhat, d)
                if self.reg_ae > 0:
                    xae = self.model(xdec)
                    loss += self.reg_ae * self.lossfunc(xdec, xae)
                return loss

            with torch.no_grad():
                self.resnorm[i] = np.sqrt(d.shape[0] * self.lossfunc(inv()[0], d).item())
                self.errnorm[i] = np.sqrt(np.prod(len(mtrue.view(-1))) * float(
                    self.lossfunc(self.model_rec(inv.x)[0],
                                  mtrue.view(-1)).item()))

            # perform line search step
            options = {'closure': closure, 'current_loss': obj,
                       'eta': 2, 'max_ls': max_ls,
                       'interpolate': interpolate,
                       'inplace': False}
            if (line_search == 'Armijo'):
                obj, lr, backtracks, clos_evals, desc_dir, fail = \
                    optimizer.step(options=options)
                # compute gradient at new iterate
                obj.backward()
                grad = optimizer._gather_flat_grad()

            elif (line_search == 'Wolfe'):
                obj, grad, lr, backtracks, clos_evals, grad_evals, desc_dir, fail = \
                    optimizer.step(options=options)

            x_new.copy_(inv.x)
            self.func_evals += clos_evals
            self.resnorm[i] = obj.item()

            # compute quantities for checking convergence
            grad_norm = torch.norm(grad)
            x_dist = torch.norm(x_new - x_old) / torch.norm(x_old)
            #x_dist[x_dist == np.inf] = 1e10
            f_dist = torch.abs(obj - f_old) / torch.max(torch.tensor(1, dtype=torch.float, device=self.device), torch.abs(f_old))

            # print log
            if verb:
                print(' %6g      |  %.3e  |  %.3e  |   %.3e    |  %.3e  |  %.3e  ' %
                      (i + 1, obj.item(), grad_norm.item(), x_dist.item(),
                       clos_evals, lr))

            # stopping criteria
            if fail or torch.isnan(obj) or i == self.niter - 1:
                break
            if (torch.norm(grad) < tol or x_dist < 1e-4 or f_dist < 1e-7 or obj.item() == -float('inf')):
                break
            x_old.copy_(x_new)
            f_old.copy_(obj)

        return self.model_rec(inv.x)[0].cpu().detach().numpy(), \
               inv.x.cpu().detach().numpy().squeeze()
