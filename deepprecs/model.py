import torch
import torch.nn as nn


def act(act_fun='LeakyReLU'):
    """Activation function

    Easy selection of activation functions by passing string or module (e.g. nn.ReLU)

    Parameters
    ----------
    act_fun : :obj:`str` or :obj:`torch.nn`
        Activation function name or function signature

    Returns
    -------
    act_fun : :obj:`torch.nn`
        Activation function signature

    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        else:
            assert False
    else:
        return act_fun()


def downsample(stride=2, downsample_mode='max'):
    """Downsample operator

    Create a downsampling layer

    Parameters
    ----------
    stride : :obj:`int`
        Stride of downsampling operation
    downsample_mode : :obj:`str`
        Mode of downsampling operation (``avg`` or ``max``)

    Returns
    -------
    downsampler : :obj:`torch.nn`
        Downsampling layer

    """
    if downsample_mode == 'avg':
        downsampler = nn.AvgPool2d(stride, stride)
    elif downsample_mode == 'max':
        downsampler = nn.MaxPool2d(stride, stride)
    else:
        assert False
    return  downsampler


def Conv2d_Block(in_f, out_f, kernel_size, stride=1, bias=True, bnorm=True,
                 act_fun='LeakyReLU', dropout=None):
    """2D Convolutional Block

    Create 2D Convolutional Block composed of convolution layer,
    dropout (optional), batch normalization (optional), and activation

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    bnorm : :obj:`int`, optional
        Apply batch normalization
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)

    Returns
    -------
    block : :obj:`torch.nn`
        2D Convolutional Block

    """
    to_pad = int((kernel_size - 1) / 2) # to mantain input size

    block = [nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias), ]
    if dropout is not None:
        block.append(nn.Dropout(dropout))
    if bnorm:
        block = block + [nn.BatchNorm2d(out_f),]
    if act_fun is not None:
        block = block + [act(act_fun), ]
    block = nn.Sequential(*block)
    return block


def ConvTranspose2d_Block(in_f, out_f, kernel_size, stride=2, bias=True, bnorm=True,
                          act_fun='LeakyReLU', dropout=None):
    """2D Transpose Convolutional Block

    Create 2D Transpose Convolutional Block composed of transpose convolution layer,
    dropout (optional), batch normalization (optional), and activation

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    bnorm : :obj:`int`, optional
        Apply batch normalization
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)

    Returns
    -------
    block : :obj:`torch.nn`
        2D Transpose Convolutional Block

    """
    block = [nn.ConvTranspose2d(in_f, out_f, kernel_size, stride, bias=bias), ]
    if dropout is not None:
        block.append(nn.Dropout(dropout))
    if bnorm:
        block = block + [nn.BatchNorm2d(out_f), ]
    if act_fun is not None:
        block = block + [act(act_fun), ]
    block = nn.Sequential(*block)
    return block


def UpsampleConv2d_Block(in_f, out_f, kernel_size, stride=2, bias=True, bnorm=True,
                         act_fun='LeakyReLU', dropout=None):
    """2D Upsampling Convolutional Block

    Create 2D Upsample Convolutional Block composed of upsampling layer,
    convolutional layer, dropout (optional), batch normalization (optional),
    and activation

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    bnorm : :obj:`int`, optional
        Apply batch normalization
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)

    Returns
    -------
    block : :obj:`torch.nn`
        2D Upsampling Convolutional Block

    """
    to_pad = int((kernel_size - 1) / 2)

    block = [nn.UpsamplingBilinear2d(scale_factor=stride),
             nn.Conv2d(in_f, out_f, kernel_size, 1, padding=to_pad, bias=bias)]
    if dropout is not None:
        block.append(nn.Dropout(dropout))
    if bnorm:
        block = block + [nn.BatchNorm2d(out_f), act(act_fun)]
    block = nn.Sequential(*block)
    return block


def Upsample1DConv2d_Block(in_f, out_f, kernel_size, stride=2, bias=True, bnorm=True,
                         act_fun='LeakyReLU', dropout=None):
    """1D Upsampling Convolutional Block

    Create 1D Upsample Convolutional Block composed of upsampling layer,
    convolutional layer, dropout (optional), batch normalization (optional),
    and activation. Here upsampling is only performed on the first dimension
    of the 2D input.

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    bnorm : :obj:`int`, optional
        Apply batch normalization
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)

    Returns
    -------
    block : :obj:`torch.nn`
        1D Upsampling Convolutional Block

    """
    to_pad = int((kernel_size - 1) / 2)

    block = [nn.Upsample(scale_factor=(2, 1), mode='bilinear'),
             nn.Conv2d(in_f, out_f, kernel_size, 1, padding=to_pad, bias=bias)]
    if dropout is not None:
        block.append(nn.Dropout(dropout))
    if bnorm:
        block = block + [nn.BatchNorm2d(out_f), act(act_fun)]
    block = nn.Sequential(*block)
    return block


class ResNetBlock(nn.Module):
    """Residual Block

    Create Residual Block (See https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)
    composed of two 2D convolutional blocks and a skip connection.

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Stride
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    expansion : :obj:`bool`, optional
        Multiplicative factor to the number of output channels (can be used to both increase or reduce
        the overall number of output channels

    Returns
    -------
    block : :obj:`torch.nn`
        Residual Block

    """
    def __init__(self, in_f, out_f, kernel_size, stride=1,
                 act_fun='LeakyReLU', expansion=1):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2d_Block(in_f, out_f, kernel_size, stride=stride,
                         bias=False, act_fun=act_fun),
            Conv2d_Block(out_f, out_f * expansion, kernel_size,
                         bias=False, act_fun=None),
        )

        self.shortcut = Conv2d_Block(in_f, out_f * expansion, kernel_size=1,
                                     stride=stride, bias=False, bnorm=True,
                                     act_fun=None)
        self.accfun = act(act_fun)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.accfun(x)
        return x


class MultiResBlock(nn.Module):
    """Multiresolution Block

    Create Multiresolution Block (See https://github.com/polimi-ispl/deep_prior_interpolation/)
    composed of three concatenated multi-scale 2D convolutional blocks and skip connection.

    Parameters
    ----------
    U : :obj:`int`
        First parameter defining the size of each convolutional block
    f_in : :obj:`int`
        Input size
    alpha : :obj:`float`, optional
        Second parameter defining the size of each convolutional block
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    bias : :obj:`bool`, optional
        Add bias to convolution layer

    Returns
    -------
    block : :obj:`torch.nn`
        Multiresolution Block

    """
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        super(MultiResBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = Conv2d_Block(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                     bias=bias, act_fun=act_fun)
        self.conv3x3 = Conv2d_Block(f_in, int(W * 0.167), 3, 1, bias=bias,
                                    act_fun=act_fun)
        self.conv5x5 = Conv2d_Block(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                    act_fun=act_fun)
        self.conv7x7 = Conv2d_Block(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                    act_fun=act_fun)
        self.bn1 = nn.BatchNorm2d(self.out_dim)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.accfun = act(act_fun)

    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], axis=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


def Conv2d_ChainOfLayers(in_f, kernel_size, nfilts, nlayers,
                         stride=1, bias=True, act_fun='LeakyReLU',
                         dropout=None, downstride=2, downmode='max'):
    """Stack of 2D Convolutional layers followed by downsampling

    Create a stack of 2D Convolutional layers followed by downsampling

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    kernel_size : :obj:`int`
        Kernel size
    nfilts : :obj:`int`
        Number of filters per layer (also output size)
    nlayers : :obj:`int`
        Number of layers
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    downstride : :obj:`int`
        Stride of downsampling operation
    downmode : :obj:`str`
        Mode of downsampling operation (``avg`` or ``max``)

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    conv_layers = []
    for ilayer in range(nlayers):
        conv_layers.append(Conv2d_Block(in_f if ilayer == 0 else nfilts,
                                        nfilts, kernel_size, stride=stride,
                                        bias=bias, act_fun=act_fun,
                                        dropout=dropout))
    downsamples = downsample(stride=downstride, downsample_mode=downmode)
    model = nn.Sequential(*(conv_layers + [downsamples]))
    return model


def ConvTranspose2d_ChainOfLayers(in_f, kernel_size, nfilts, nlayers,
                                  stride=1, bias=True, act_fun='LeakyReLU',
                                  dropout=None, upstride=2, upmode='convtransp'):
    """Stack of 2D Convolutional layers preceeded by transpose convolution or upsampling

    Create a stack of 2D Convolutional layers preceeded by transpose convolution or upsampling

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    kernel_size : :obj:`int`
        Kernel size
    nfilts : :obj:`int`
        Number of filters per layer (also output size)
    nlayers : :obj:`int`
        Number of layers
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    upstride : :obj:`int`
        Stride of upsampling operation
    upmode : :obj:`str`
        Mode of upsampling operation (``convtransp`` or ``upsample``)

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    if upmode == 'convtransp':
        convtransp = ConvTranspose2d_Block(in_f, nfilts, kernel_size-1,
                                           stride=upstride, bias=bias,
                                           act_fun=act_fun)
    elif upmode == 'upsample':
        convtransp = UpsampleConv2d_Block(in_f, nfilts, kernel_size,
                                          stride=upstride, bias=bias,
                                          act_fun=act_fun, dropout=dropout)
        
    conv_layers = [convtransp, ]
    for ilayer in range(nlayers):
        conv_layers.append(Conv2d_Block(nfilts, nfilts,
                                        kernel_size, stride=stride,
                                        bias=True, act_fun=act_fun, 
                                        dropout=dropout))

    model = nn.Sequential(*conv_layers)
    return model


def ConvTranspose2d_ChainOfLayers1(in_f, kernel_size, nfilts, nlayers,
                                   stride=1, bias=True,
                                   act_fun='LeakyReLU', dropout=None,
                                   upstride=2, upmode='convtransp'):
    """Stack of 2D Convolutional layers preceeded by transpose convolution or upsampling

    Create a stack of 2D Convolutional layers preceeded by transpose convolution or upsampling
    Note that the number of layers is changed at the last step instead of the first as done in the
    `ConvTranspose2d_ChainOfLayers` module.

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    kernel_size : :obj:`int`
        Kernel size
    nfilts : :obj:`int`
        Number of filters per layer (also output size)
    nlayers : :obj:`int`
        Number of layers
    stride : :obj:`int`, optional
        Stride
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    dropout : :obj:`float`, optional
        Percentage of dropout (if ``None``, dropout is not applied)
    upstride : :obj:`int`
        Stride of upsampling operation
    upmode : :obj:`str`
        Mode of upsampling operation (``convtransp`` or ``upsample``)

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    if upmode == 'convtransp':
        convtransp = ConvTranspose2d_Block(in_f, in_f, kernel_size - 1,
                                           stride=upstride, bias=bias,
                                           act_fun=act_fun)
    elif upmode == 'upsample':
        convtransp = UpsampleConv2d_Block(in_f, in_f, kernel_size,
                                          stride=upstride, bias=bias,
                                          act_fun=act_fun, dropout=dropout)
    elif upmode == 'upsample1d':
        convtransp = Upsample1DConv2d_Block(in_f, in_f, kernel_size,
                                           stride=upstride, bias=bias,
                                           act_fun=act_fun, dropout=dropout)

    conv_layers = [convtransp, ]
    for ilayer in range(nlayers):
        conv_layers.append(
            Conv2d_Block(in_f, in_f if ilayer < nlayers - 1 else nfilts,
                         kernel_size, stride=stride,
                         bias=bias, act_fun=act_fun,
                         dropout=dropout))

    model = nn.Sequential(*conv_layers)
    return model


def ResNet_Layer(in_f, out_f, kernel_size, act_fun='LeakyReLU',
                 expansion=1, nlayers=1, downstride=2, downmode='max'):
    """ResNet Layer

    Create a ResNet layer composed by multiple ResNet blocks blocks stacked one after the other,
    optionally followed by a downsampling layer.

    Note that if in_f and out_f are different the width and height are automatically downsampled.
    Expansion allow to go from in_f to out_f*expansion whilst keeping the intermediate layer to
    out_f (and it is only applied to the first layer).

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    expansion : :obj:`bool`, optional
        Multiplicative factor to the number of output channels (can be used to both increase or reduce
        the overall number of output channels
    nlayers : :obj:`int`
        Number of layers
    downstride : :obj:`int`, optional
        Stride of downsampling operation (if 1, downsampling is not added)
    downmode : :obj:`str`
        Mode of downsampling operation (``avg`` or ``max``)

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    stride = 2 if in_f != out_f else 1
    res = nn.Sequential(
        ResNetBlock(in_f, out_f, kernel_size, act_fun=act_fun,
                    stride=stride, expansion=expansion),
        *[ResNetBlock(out_f * expansion, out_f * expansion, kernel_size,
                      act_fun=act_fun, stride=1, expansion=1)
          for _ in range(nlayers - 1)])

    if downstride > 1:
        downsamples = downsample(stride=downstride, downsample_mode=downmode)
        res = [res, downsamples]

    model = nn.Sequential(*res)
    return model


def ResNetTranspose_Layer(in_f, out_f, kernel_size, act_fun='LeakyReLU',
                          expansion=1, nlayers=1, upstride=2):
    """ResNet Layer

    Create a ResNet layer composed by multiple ResNet blocks blocks stacked one after,
    preceeded by an upsampling layer.

    Note that if in_f and out_f are different the width and height are automatically downsampled.
    Expansion allow to go from in_f to out_f*expansion whilst keeping the intermediate layer to
    out_f (and it is only applied to the first layer)

    Parameters
    ----------
    in_f : :obj:`int`
        Input size
    out_f : :obj:`int`
        Output size
    kernel_size : :obj:`int`
        Kernel size
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    nlayers : :obj:`int`
        Number of layers
    upstride : :obj:`int`
        Stride of upsampling operation

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    upsamples = nn.UpsamplingBilinear2d(scale_factor=upstride)

    stride = 1 # ensure there is no downsampling
    res = nn.Sequential(
        ResNetBlock(in_f, out_f, kernel_size, act_fun=act_fun,
                    stride=stride, expansion=expansion),
        *[ResNetBlock(out_f * expansion, out_f * expansion, kernel_size,
                      act_fun=act_fun, stride=1, expansion=1)
          for _ in range(nlayers - 1)])

    conv_layers = [upsamples, res]

    model = nn.Sequential(*conv_layers)
    return model


def MultiRes_Layer(U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True,
                   downstride=2, downmode='max'):
    """Multires Layer with downsampling

    Create a Multires layer followed by a downsampling layer.

    Parameters
    ----------
    U : :obj:`int`
        First parameter defining the size of each convolutional block
    f_in : :obj:`int`
        Input size
    alpha : :obj:`float`, optional
        Second parameter defining the size of each convolutional block
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    downstride : :obj:`int`, optional
        Stride of downsampling operation
    downmode : :obj:`str`
        Mode of downsampling operation (``avg`` or ``max``)

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    mres = MultiResBlock(U, f_in, alpha=alpha, act_fun=act_fun, bias=bias)
    out_dim = mres.out_dim
    downsamples = downsample(stride=downstride, downsample_mode=downmode)
    conv_layers = [mres, downsamples]

    model = nn.Sequential(*conv_layers)
    return model, out_dim


def MultiResTranspose_Layer(U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True,
                            upstride=2):
    """Multires Layer with upsampling

    Create a Multires layer preceeded by an upsampling layer.

    Parameters
    ----------
    U : :obj:`int`
        First parameter defining the size of each convolutional block
    f_in : :obj:`int`
        Input size
    alpha : :obj:`float`, optional
        Second parameter defining the size of each convolutional block
    act_fun : :obj:`str` or :obj:`torch.nn`, optional
        Activation function name or function signature
    bias : :obj:`bool`, optional
        Add bias to convolution layer
    upstride : :obj:`int`, optional
        Stride of upstride operation

    Returns
    -------
    model : :obj:`torch.nn`
        Model

    """
    upsamples = nn.UpsamplingBilinear2d(scale_factor=upstride)
    mres = MultiResBlock(U, f_in, alpha=alpha, act_fun=act_fun, bias=bias)
    out_dim = mres.out_dim
    conv_layers = [upsamples, mres]

    model = nn.Sequential(*conv_layers)
    return model, out_dim