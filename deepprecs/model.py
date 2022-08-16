import numpy as np
import torch
import torch.nn as nn


def act(act_fun='LeakyReLU'):
    """Easy selection of activation function by passing string or
    module (e.g. nn.ReLU)
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
    if downsample_mode == 'avg':
        downsampler = nn.AvgPool2d(stride, stride)
    elif downsample_mode == 'max':
        downsampler = nn.MaxPool2d(stride, stride)
    else:
        assert False
    return  downsampler


def Conv2d_Block(in_f, out_f, kernel_size, stride=1, bias=True, bnorm=True,
                 act_fun='LeakyReLU', dropout=None):
    """2d Convolutional Block (conv, dropout, batchnorm, activation)
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
    """2d Transpose Convolutional Block (transpconv, batchnorm, activation)
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
    """2d Upsampling and Convolutional Block (upsample, conv, batchnorm, activation)
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
    """1d Upsampling along first direction and Convolutional Block (upsample, conv, batchnorm, activation)
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
    def __init__(self, in_f, out_f, kernel_size, stride=1, act_fun='LeakyReLU',
                 expansion=1):
        """Residual Block (See https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)
        """
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
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        """Multiresolution block (See https://github.com/polimi-ispl/deep_prior_interpolation/)
        """
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
    """2d Convolutional Block with multiple convolution layers followed by
    2d Downsampling
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
    """2d Transpose Convolutional Block with multiple convolution layers 
    (first is ConvTranspose2d and others are Conv2d)
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
    """2d Transpose Convolutional Block with multiple convolution layers
    (first is ConvTranspose2d and others are Conv2d).

    Note that the number of layers is changed at the last step instead
    of the first as done in the ConvTranspose2d_ChainOfLayers module
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
    """
    A ResNet layer composed by nlayers blocks stacked one after the other

    Note that if in_f and out_f are different the width and height are
    automatically downsampled. Expansion allow to go from in_f to out_f*expansion
    whilst keeping the intermediate layer to out_f (and it is only applied to
    the first layer)
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
    """
    A ResNet layer composed by nlayers blocks stacked one after the other
    preceeed by upsampling
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
    mres = MultiResBlock(U, f_in, alpha=alpha, act_fun=act_fun, bias=bias)
    out_dim = mres.out_dim
    downsamples = downsample(stride=downstride, downsample_mode=downmode)
    conv_layers = [mres, downsamples]

    model = nn.Sequential(*conv_layers)
    return model, out_dim


def MultiResTranspose_Layer(U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True,
                            upstride=2):
    upsamples = nn.UpsamplingBilinear2d(scale_factor=upstride)
    mres = MultiResBlock(U, f_in, alpha=alpha, act_fun=act_fun, bias=bias)
    out_dim = mres.out_dim
    conv_layers = [upsamples, mres]

    model = nn.Sequential(*conv_layers)
    return model, out_dim


class Autoencoder(nn.Module):
    """Template AutoEncoder module
    """
    def __init__(self):
        super(Autoencoder, self).__init__()

    def encode(self, x):
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

    def restricted_decode(self, x):
        x = self.decode(x)
        x = x.view([-1])
        x = self.restriction.apply(x)
        return x

    def patched_restricted_decode(self, x):
        x = x.view([self.npatches, self.nenc])
        xdec = self.decode(x)
        #if self.superres:
        #    xdec = xdec[:, :, ::2]
        x = self.patchesscaling * xdec
        x = x.view([-1])
        x = self.patcher.apply(x)
        x = self.restriction.apply(x)
        return x, xdec

    def patched_decode(self, x):
        x = x.view([self.npatches, self.nenc])
        xdec = self.decode(x)
        x = self.patchesscaling * xdec
        x = x.view([-1])
        x = self.patcher.apply(x)
        return x, xdec

    def patched_forward(self, x, intermediate=False):
        # make patches
        x = self.depatcher.apply(x)
        x = x.view([self.npatches, 1, self.nh, self.nw])
        xsc = x / self.patchesscaling
        # pass through network
        xdec = self.forward(xsc)
        # make back into unique data
        x = self.patchesscaling * xdec
        x = x.view([-1])
        x = self.patcher.apply(x)
        if not intermediate:
            return x
        else:
            return x, xsc, xdec

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class AutoencoderBase(Autoencoder):
    """Base AutoEncoder module
    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels,
                 restriction,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, depatcher=None, npatches=None, patchesscaling=None,
                 superres=False, conv11=False, conv11size=1):
        super(AutoencoderBase, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.restriction = restriction
        self.patcher = patcher
        self.depatcher = depatcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.superres = superres
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
                                     bias=convbias, bnorm=bnormlast,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2 * conv11size, 2 * nfilts[-1],
                                     1, stride=1,
                                     bias=convbias, bnorm=bnormlast,
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
        # self.nfiltslatent = nfilts[-1]
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
        self.restriction = restriction


class AutoencoderSymmetric(Autoencoder):
    """AutoEncoder module with symmetry between encoder and decoder
    (AutoencoderBase has some asymmetry)
    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels,
                 restriction,
                 convbias=True, act_fun='LeakyReLU', dropout=None, downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False, tanh_final=False,
                 patcher=None, depatcher=None, npatches=None, patchesscaling=None,
                 superres=False):
        super(AutoencoderSymmetric, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.restriction = restriction
        self.patcher = patcher
        self.depatcher = depatcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.superres = superres
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
        #self.nfiltslatent = nfilts[-1]
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
        if superres:
            conv_blocks.append(ConvTranspose2d_ChainOfLayers1(nfilts[0], kernel_size=kernel_size[0],
                                                              nfilts=nfilts[0],
                                                              nlayers=nlayers-2,
                                                              stride=1, bias=convbias,
                                                              act_fun=act_fun,
                                                              dropout=dropout,
                                                              upstride=downstride,
                                                              upmode='upsample1d'))
        conv_blocks.append(Conv2d_Block(nfilts[0], 1,
                                        kernel_size[0], stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)
        self.restriction = restriction

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
    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels,
                 restriction,
                 convbias=True, act_fun='LeakyReLU', dropout=None,
                 downstride=1,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False,
                 tanh_final=False,
                 patcher=None, depatcher=None, npatches=None, patchesscaling=None,
                 superres=False, conv11=False, conv11size=1):
        super(AutoencoderRes, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.restriction = restriction
        self.patcher = patcher
        self.depatcher = depatcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.superres = superres
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
                                     bias=convbias, bnorm=bnormlast,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2, 2 * nfilts[-1],
                                     1, stride=1,
                                     bias=convbias, bnorm=bnormlast,
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
        if superres:
            conv_blocks.append(ConvTranspose2d_ChainOfLayers1(nfilts[0], kernel_size=kernel_size[0],
                                                              nfilts=nfilts[0],
                                                              nlayers=nlayers - 2,
                                                              stride=1, bias=convbias,
                                                              act_fun=act_fun,
                                                              dropout=dropout,
                                                              upstride=downstride,
                                                              upmode='upsample1d'))
        conv_blocks.append(Conv2d_Block(nfilts[0], 1,
                                        kernel_size[0], stride=1,
                                        bias=convbias, bnorm=bnormlast,
                                        act_fun=None,
                                        dropout=dropout))
        self.dec = nn.Sequential(*conv_blocks)
        self.restriction = restriction


class AutoencoderMultiRes(Autoencoder):
    """MultiResolution AutoEncoder module
    """
    def __init__(self, nh, nw, nenc, kernel_size, nfilts, nlayers, nlevels,
                 restriction,
                 convbias=True, act_fun='LeakyReLU', dropout=None,
                 downstride=2,
                 downmode='max', upmode='convtransp', bnormlast=True,
                 relu_enc=False, tanh_enc=False, relu_dec=False,
                 tanh_final=False,
                 patcher=None, depatcher=None, npatches=None, patchesscaling=None,
                 superres=False, conv11=False, conv11size=1):
        super(AutoencoderMultiRes, self).__init__()
        self.nh, self.nw = nh, nw
        self.nhlatent = nh // (2 ** nlevels)
        self.nwlatent = nw // (2 ** nlevels)
        self.nenc = nenc
        self.restriction = restriction
        self.patcher = patcher
        self.depatcher = depatcher
        self.npatches = npatches
        self.patchesscaling = patchesscaling
        self.reluflag_enc = relu_enc
        self.tanhflag_enc = tanh_enc
        self.reluflag_dec = relu_dec
        self.tanhflag = tanh_final
        self.superres = superres
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
                                     bias=convbias, bnorm=bnormlast,
                                     act_fun=act_fun,
                                     dropout=dropout)
            self.c11d = Conv2d_Block(2 * conv11size, 2 * in_f,
                                     1, stride=1,
                                     bias=convbias, bnorm=bnormlast,
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
        self.restriction = restriction

