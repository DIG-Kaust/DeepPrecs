import torch
import torch.nn as nn

from deepprecs.model import *


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

