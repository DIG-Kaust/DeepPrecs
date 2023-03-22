from deepprecs.model import *


class Autoencoder(nn.Module):
    """Template AutoEncoder network

    Create and apply AutoEncoder network. This class is a template
    which cannot be run directly; users are instead required to subclass it
    and provide a number of required inputs - see :class:`deepprecs.aemodel.AutoencoderBase`
    for an example.

    """
    def __init__(self):
        super(Autoencoder, self).__init__()

    def encode(self, x):
        """Encoder path

        Apply encoder

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output

        """
        x = self.enc(x)
        if self.conv11:
            x = self.c11e(x)
        x = x.view([x.size(0), self.nfiltslatent * self.nhlatent * self.nwlatent])
        x = self.le(x)
        if self.reluflag_enc:
            x = act('ReLU')(x)
        elif self.tanhflag_enc:
            x = act('Tanh')(x)
        return x

    def decode(self, x):
        """Decoder path

        Apply decoder

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output

        """
        x = self.ld(x)
        if self.reluflag_dec:
            x = act('ReLU')(x)
        x = x.view([x.size(0), 2 * self.nfiltslatent, self.nhlatent, self.nwlatent])
        if self.conv11:
            x = self.c11d(x)
        x = self.dec(x)
        if self.tanhflag:
            x = self.tanh(x)
        return x

    def forward(self, x):
        """Autoencoder path

        Apply autoencoder

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output

        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def physics_decode(self, x):
        """Decoder path with physics-based operator

        Apply decoder followed by physics-based operator

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output

        """
        x = self.decode(x)
        x = x.view([-1])
        x = self.physics.apply(x)
        return x

    def patched_decode(self, x):
        """Decoder path with patching

        Apply decoder followed by patching

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output
        xdec : :obj:`torch.tensor`
            Decoded output (prior to repatching)

        """
        x = x.view([self.npatches, self.nenc])
        xdec = self.decode(x)
        x = self.patchesscaling * xdec
        if self.patchesshift is not None:
            x = x + self.patchesshift
        x = x.view([-1])
        x = self.patcher.apply(x)
        return x, xdec

    def patched_physics_decode(self, x):
        """Decoder path with patching and physics-based operator

        Apply decoder followed by patching and physics-based operator

        Parameters
        ----------
        x : :obj:`torch.tensor`
            Input

        Returns
        -------
        x : :obj:`torch.tensor`
            Output
        xdec : :obj:`torch.tensor`
            Decoded output (prior to repatching and physics-based operator)

        """
        x, xdec = self.patched_decode(x)
        x = self.physics.apply(x)
        return x, xdec


class AutoencoderBase(Autoencoder):
    """Base AutoEncoder module

    Create and apply purely convolutional AutoEncoder network.

    Parameters
    ----------
    nh : :obj:`int`
        Height of input images
    nw : :obj:`torch.nn.Module`
        Width of input images
    nenc : :obj:`int`
        Size of latent code
    kernel_size : :obj:`int` or :obj:`list`
        Kernel size (constant for all levels, or different for each level)
    nfilts : :obj:`int` or :obj:`list`
        Number of filters per layer (constant for all levels, or different for each level)
    nlayers : :obj:`int`
        Number of layers per level
    nlayers : :obj:`int`
        Number of levels
    physics : :obj:`pylops_gpu.TorchOperator`
        Physical operator
    convbias : :obj:`bool`, optional
        Add bias to convolution layers
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    downstride : :obj:`int`, optional
        Stride of downsampling operation
    downmode : :obj:`str`, optional
        Mode of downsampling operation (``avg`` or ``max``)
    upmode : :obj:`str`, optional
        Mode of upsampling operation (``convtransp`` or ``upsample``)
    bnormlast : :obj:`bool`, optional
        Apply batch normalization to last convolutional layer of decoder
    relu_enc : :obj:`bool`, optional
        Apply ReLU activation to linear layer of encoder
    tanh_enc : :obj:`bool`, optional
        Apply Tanh activation to linear layer of encoder
    relu_dec : :obj:`bool`, optional
        Apply ReLU activation to linear layer of decoder
    tanh_final : :obj:`bool`, optional
        Apply Tanh activation to final layer of decoder
    patcher : :obj:`pylops_gpu.TorchOperator`, optional
        Patching operator
    npatches : :obj:`int`, optional
        Number of patches (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    patchesscaling : :obj:`torch.tensor`, optional
        Scalings to apply to each patch (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    conv11 : :obj:`bool`, optional
        Apply 1x1 convolution at the bottleneck before linear layer of encoder and after linear layer of decoder
    conv11size : :obj:`int`, optional
        Number of output channels of 1x1 convolution layer
    patchesshift : :obj:`float`, optional
        Shift to apply to each patch (optional when using :func:`deepprecs.model.Autoencoder.patched_decode` method)

    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels, physics,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, npatches=None, patchesscaling=None, conv11=False, conv11size=1,
                 patchesshift=None):
        super(AutoencoderBase, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.physics = physics
        self.patcher = patcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.patchesshift = patchesshift
        self.patchescaling = patchescaling
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.conv11 = conv11
        self.conv11size = conv11size # force to reduce to a user-defined number of channels (1 as default)

        # define kernel sizes
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * nlevels

        # encoder convolutions
        nfilts = [1, ] + [nfilts * (2 ** i) for i in range(nlevels)]
        conv_blocks = [Conv2d_ChainOfLayers(in_f, kernel_size=ks,
                                            nfilts=out_f, nlayers=nlayers,
                                            stride=1, bias=convbias,
                                            act_fun=act_fun,
                                            dropout=dropout,
                                            downstride=downstride,
                                            downmode=downmode)
                       for in_f, out_f, ks in zip(nfilts, nfilts[1:], kernel_size)]
        self.enc = nn.Sequential(*conv_blocks)

        # conv 1x1 layers
        if self.conv11:
            self.c11e = Conv2d_Block(nfilts[-1], conv11size,
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2 * conv11size, 2 * nfilts[-1],
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)

        # dense layers
        if self.conv11:
            self.nfiltslatent = conv11size
        else:
            self.nfiltslatent = nfilts[-1]
        self.le = nn.Linear(self.nfiltslatent * self.nhlatent * self.nwlatent, nenc)
        self.ld = nn.Linear(nenc, (2 * self.nfiltslatent) * self.nhlatent * self.nwlatent)
        self.tanh = nn.Tanh()
        
        # decoder convolutions
        nfilts = nfilts[1:] + [nfilts[-1] * 2, ]
        conv_blocks = [ConvTranspose2d_ChainOfLayers(in_f, kernel_size=ks,
                                                     nfilts=out_f, nlayers=nlayers,
                                                     stride=1, bias=convbias,
                                                     act_fun=act_fun,
                                                     dropout=dropout,
                                                     upstride=downstride,
                                                     upmode=upmode)
                       for in_f, out_f, ks in zip(nfilts[::-1], nfilts[::-1][1:], kernel_size[::-1])]
        conv_blocks.append(Conv2d_Block(nfilts[0], 1,
                                        kernel_size[0], stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)


class AutoencoderSymmetric(Autoencoder):
    """Symmetric AutoEncoder module

    Create and apply purely convolutional AutoEncoder network with symmetric encoder and decoder
    (AutoencoderBase has some asymmetry)

    Parameters
    ----------
    nh : :obj:`int`
        Height of input images
    nw : :obj:`torch.nn.Module`
        Width of input images
    nenc : :obj:`int`
        Size of latent code
    kernel_size : :obj:`int` or :obj:`list`
        Kernel size (constant for all levels, or different for each level)
    nfilts : :obj:`int` or :obj:`list`
        Number of filters per layer (constant for all levels, or different for each level)
    nlayers : :obj:`int`
        Number of layers per level
    nlayers : :obj:`int`
        Number of levels
    physics : :obj:`pylops_gpu.TorchOperator`
        Physical operator
    convbias : :obj:`bool`, optional
        Add bias to convolution layers
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    downstride : :obj:`int`, optional
        Stride of downsampling operation
    downmode : :obj:`str`, optional
        Mode of downsampling operation (``avg`` or ``max``)
    upmode : :obj:`str`, optional
        Mode of upsampling operation (``convtransp`` or ``upsample``)
    bnormlast : :obj:`bool`, optional
        Apply batch normalization to last convolutional layer of decoder
    relu_enc : :obj:`bool`, optional
        Apply ReLU activation to linear layer of encoder
    tanh_enc : :obj:`bool`, optional
        Apply Tanh activation to linear layer of encoder
    relu_dec : :obj:`bool`, optional
        Apply ReLU activation to linear layer of decoder
    tanh_final : :obj:`bool`, optional
        Apply Tanh activation to final layer of decoder
    patcher : :obj:`pylops_gpu.TorchOperator`, optional
        Patching operator
    npatches : :obj:`int`, optional
        Number of patches (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    patchesscaling : :obj:`torch.tensor`, optional
        Scalings to apply to each patch (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    patchesshift : :obj:`float`, optional
        Shift to apply to each patch (optional when using :func:`deepprecs.model.Autoencoder.patched_decode` method)

    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels, physics,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, npatches=None, patchesscaling=None, patchesshift=None):
        super(AutoencoderSymmetric, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.physics = physics
        self.patcher = patcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.patchesshift = patchesshift
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.conv11 = False

        # define kernel sizes
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * nlevels

        # encoder convolutions
        nfilts = [1, ] + [nfilts * (2 ** i) for i in range(nlevels)]
        self.nfiltslatent = nfilts[-1]
        conv_blocks = [Conv2d_ChainOfLayers(in_f, kernel_size=ks,
                                            nfilts=out_f, nlayers=nlayers,
                                            stride=1, bias=convbias,
                                            act_fun=act_fun,
                                            dropout=dropout,
                                            downstride=downstride,
                                            downmode=downmode)
                        for in_f, out_f, ks in zip(nfilts, nfilts[1:], kernel_size)]

        self.enc = nn.Sequential(*conv_blocks)

        # dense layers
        self.le = nn.Linear(self.nfiltslatent * self.nhlatent * self.nwlatent, nenc)
        self.ld = nn.Linear(nenc, self.nfiltslatent * self.nhlatent * self.nwlatent)
        self.tanh = nn.Tanh()

        # decoder convolutions
        nfilts = [nfilts[1]] + nfilts[1:]
        conv_blocks = [ConvTranspose2d_ChainOfLayers1(in_f, kernel_size=ks,
                                                      nfilts=out_f,
                                                      nlayers=nlayers-1 if iblock < len(nfilts)-2 else nlayers-2,
                                                      stride=1, bias=convbias,
                                                      act_fun=act_fun,
                                                      dropout=dropout,
                                                      upstride=downstride,
                                                      upmode=upmode)
                       for iblock, (in_f, out_f, ks) in enumerate(zip(nfilts[::-1], nfilts[::-1][1:], kernel_size[::-1]))]

        conv_blocks.append(Conv2d_Block(nfilts[0], 1,
                                        kernel_size[0], stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)

    def decode(self, x):
        x = self.ld(x)
        if self.reluflag_dec:
            x = act('ReLU')(x)
        x = x.view([x.size(0), self.nfiltslatent, self.nhlatent, self.nwlatent])
        x = self.dec(x)
        if self.tanhflag:
            x = self.tanh(x)
        return x


class AutoencoderRes(Autoencoder):
    """ResNet AutoEncoder module

    Create and apply AutoEncoder network with ResNet blocks

    Parameters
    ----------
    nh : :obj:`int`
        Height of input images
    nw : :obj:`torch.nn.Module`
        Width of input images
    nenc : :obj:`int`
        Size of latent code
    kernel_size : :obj:`int` or :obj:`list`
        Kernel size (constant for all levels, or different for each level)
    nfilts : :obj:`int` or :obj:`list`
        Number of filters per layer (constant for all levels, or different for each level)
    nlayers : :obj:`int`
        Number of layers per level
    nlayers : :obj:`int`
        Number of levels
    physics : :obj:`pylops_gpu.TorchOperator`
        Physical operator
    convbias : :obj:`bool`, optional
        Add bias to convolution layers
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    downstride : :obj:`int`, optional
        Stride of downsampling operation
    downmode : :obj:`str`, optional
        Mode of downsampling operation (``avg`` or ``max``)
    upmode : :obj:`str`, optional
        Mode of upsampling operation (``convtransp`` or ``upsample``)
    bnormlast : :obj:`bool`, optional
        Apply batch normalization to last convolutional layer of decoder
    relu_enc : :obj:`bool`, optional
        Apply ReLU activation to linear layer of encoder
    tanh_enc : :obj:`bool`, optional
        Apply Tanh activation to linear layer of encoder
    relu_dec : :obj:`bool`, optional
        Apply ReLU activation to linear layer of decoder
    tanh_final : :obj:`bool`, optional
        Apply Tanh activation to final layer of decoder
    patcher : :obj:`pylops_gpu.TorchOperator`, optional
        Patching operator
    npatches : :obj:`int`, optional
        Number of patches (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    patchesscaling : :obj:`torch.tensor`, optional
        Scalings to apply to each patch (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    conv11 : :obj:`bool`, optional
        Apply 1x1 convolution at the bottleneck before linear layer of encoder and after linear layer of decoder
    conv11size : :obj:`int`, optional
        Number of output channels of 1x1 convolution layer
    patchesshift : :obj:`float`, optional
        Shift to apply to each patch (optional when using :func:`deepprecs.model.Autoencoder.patched_decode` method)

    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels, physics,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=1,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, npatches=None, patchesscaling=None, conv11=False, conv11size=1, patchesshift=None):
        super(AutoencoderRes, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.physics = physics
        self.patcher = patcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.patchesshift = patchesshift
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.conv11 = conv11

        # define kernel sizes
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * nlevels

        # encoder convolutions
        nfilts = [1, ] + [nfilts * (2 ** i) for i in range(nlevels)]
        self.nfiltslatent = nfilts[-1]
        conv_blocks = [ResNet_Layer(in_f, out_f,
                                    kernel_size=ks,
                                    act_fun=act_fun,
                                    expansion=1,
                                    nlayers=nlayers,
                                    downstride=downstride,
                                    downmode=downmode)
                       for in_f, out_f, ks in zip(nfilts, nfilts[1:], kernel_size)]
        self.enc = nn.Sequential(*conv_blocks)

        # conv 1x1 layers
        if self.conv11:
            self.c11e = Conv2d_Block(nfilts[-1], 1,
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2, 2 * nfilts[-1],
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)
        # dense layers
        if self.conv11:
            self.nfiltslatent = 1
        else:
            self.nfiltslatent = nfilts[-1]
        self.le = nn.Linear(self.nfiltslatent * self.nhlatent * self.nwlatent, nenc)
        self.ld = nn.Linear(nenc, (2 * self.nfiltslatent) * self.nhlatent * self.nwlatent)
        self.tanh = nn.Tanh()

        # decoder convolutions
        nfilts = nfilts[1:] + [nfilts[-1] * 2, ]
        conv_blocks = [ResNetTranspose_Layer(in_f, out_f,
                                             kernel_size=ks,
                                             act_fun=act_fun,
                                             expansion=1,
                                             nlayers=nlayers,
                                             upstride=2)
                       for in_f, out_f, ks in zip(nfilts[::-1], nfilts[::-1][1:], kernel_size[::-1])]
        conv_blocks.append(Conv2d_Block(nfilts[0], 1,
                                        kernel_size[0], stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)


class AutoencoderMultiRes(Autoencoder):
    """MultiRes AutoEncoder module

    Create and apply AutoEncoder network with MultiResolution blocks

    Parameters
    ----------
    nh : :obj:`int`
        Height of input images
    nw : :obj:`torch.nn.Module`
        Width of input images
    nenc : :obj:`int`
        Size of latent code
    kernel_size : :obj:`int` or :obj:`list`
        Kernel size (constant for all levels, or different for each level)
    nfilts : :obj:`int` or :obj:`list`
        Number of filters per layer (constant for all levels, or different for each level)
    nlayers : :obj:`int`
        Number of layers per level
    nlayers : :obj:`int`
        Number of levels
    physics : :obj:`pylops_gpu.TorchOperator`
        Physical operator
    convbias : :obj:`bool`, optional
        Add bias to convolution layers
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    downstride : :obj:`int`, optional
        Stride of downsampling operation
    downmode : :obj:`str`, optional
        Mode of downsampling operation (``avg`` or ``max``)
    upmode : :obj:`str`, optional
        Mode of upsampling operation (``convtransp`` or ``upsample``)
    bnormlast : :obj:`bool`, optional
        Apply batch normalization to last convolutional layer of decoder
    relu_enc : :obj:`bool`, optional
        Apply ReLU activation to linear layer of encoder
    tanh_enc : :obj:`bool`, optional
        Apply Tanh activation to linear layer of encoder
    relu_dec : :obj:`bool`, optional
        Apply ReLU activation to linear layer of decoder
    tanh_final : :obj:`bool`, optional
        Apply Tanh activation to final layer of decoder
    patcher : :obj:`pylops_gpu.TorchOperator`, optional
        Patching operator
    npatches : :obj:`int`, optional
        Number of patches (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    patchesscaling : :obj:`torch.tensor`, optional
        Scalings to apply to each patch (required when using :func:`deepprecs.model.Autoencoder.patched_decode` method)
    conv11 : :obj:`bool`, optional
        Apply 1x1 convolution at the bottleneck before linear layer of encoder and after linear layer of decoder
    conv11size : :obj:`int`, optional
        Number of output channels of 1x1 convolution layer
    patchesshift : :obj:`float`, optional
        Shift to apply to each patch (optional when using :func:`deepprecs.model.Autoencoder.patched_decode` method)

    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels, physics,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, npatches=None, patchesscaling=None, conv11=False, conv11size=1, patchesshift=None):
        super(AutoencoderMultiRes, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.physics = physics
        self.patcher = patcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.patchesshift = patchesshift
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.conv11 = conv11

        # encoder convolutions
        nfilts = [1, ] + [nfilts * (2 ** i) for i in range(nlevels)]
        conv_blocks = []
        in_f = 1
        for i_f, out_f in enumerate(nfilts[1:]):
            ms, in_f = MultiRes_Layer(out_f, in_f,
                                      alpha=1.67,
                                      act_fun=act_fun,
                                      downstride=downstride,
                                      downmode=downmode)
            conv_blocks.append(ms)
        self.nfiltslatent = in_f
        self.enc = nn.Sequential(*conv_blocks)

        # conv 1x1 layers
        if self.conv11:
            self.c11e = Conv2d_Block(in_f, conv11size,
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2 * conv11size, 2 * in_f,
                                     1, stride=1,
                                     bias=convbias,
                                     act_fun=act_fun,
                                     dropout=dropout)

        # dense layers
        if self.conv11:
            self.nfiltslatent = conv11size
        self.le = nn.Linear(self.nfiltslatent * self.nhlatent * self.nwlatent, nenc)
        self.ld = nn.Linear(nenc, (2 * self.nfiltslatent) * self.nhlatent * self.nwlatent)
        self.tanh = nn.Tanh()

        # decoder convolutions
        nfilts = nfilts[1:] + [2 * in_f, ]
        in_f = nfilts[-1]
        conv_blocks = []
        for out_f in nfilts[::-1][1:]:
            ms, in_f = MultiResTranspose_Layer(out_f, in_f,
                                               alpha=1.67,
                                               act_fun=act_fun,
                                               upstride=downstride)
            conv_blocks.append(ms)
        conv_blocks.append(Conv2d_Block(in_f, 1,
                                        kernel_size, stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)

