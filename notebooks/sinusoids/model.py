import torch.nn as nn


class Autoencoder(nn.Module):
    """1D Dense Autoencoder

    Parameters
    ----------
    nin : :obj:`int`
        Input size
    nenc : :obj:`int`
        Latent space size (i.e., output of encoderr
    restriction : :obj:`pylops.LinearOperator`
        Restriction operator

    Attributes
    ----------
    encode : :obj:`torch.Module`
        Encoder network
    decode : :obj:`torch.Module`
        Decoder network

    """
    def __init__(self, nin, nenc, restriction):
        super(Autoencoder,self).__init__()
        self.nin = nin
        self.nenc = nenc
        self.restriction = restriction
        self.encode = nn.Sequential(
            nn.Linear(nin, 2*nenc),
            nn.ReLU(inplace=True),
            nn.Linear(2*nenc, nenc),
            nn.Tanh(),
        )
        self.decode = nn.Sequential(
            nn.Linear(nenc, 2*nenc),
            nn.ReLU(inplace=True),
            nn.Linear(2*nenc, nin),
        )

    def forward(self,x):
        """Apply model
        """
        y = self.encode(x)
        xinv = self.decode(y)
        return xinv, y

    def restricted_decode(self, x):
        """Apply decoder and restiction operator
        """
        x = self.decode(x)
        x = self.restriction.apply(x.squeeze())
        return x