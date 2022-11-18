from abc import abstractmethod

import numpy as np

from collections import OrderedDict
from typing import List
from torch.nn import functional as F

from torch import nn
import torch
import logging

log = logging.getLogger(__name__)


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def make_decoder(dec_dims, z_dim, non_linearity):
    decoder_layers = OrderedDict()
    prev_dim = z_dim
    for i, next_dim in enumerate(dec_dims[1:], start=1):
        decoder_layers[f"dec_layer_{i}"] = nn.Linear(prev_dim, next_dim)
        decoder_layers[f"dec_activation_{i}"] = non_linearity()
        prev_dim = next_dim
    # delete last activation
    del decoder_layers[f"dec_activation_{i}"]

    decoder = nn.Sequential(decoder_layers)
    return decoder


def make_encoder(n_items: int, dims: List[int], non_linearity: callable, input_dropout: float = None,
                 stochastic: bool = False) -> nn.Sequential:
    encoder_layers = OrderedDict()

    if input_dropout is not None:
        encoder_layers["input_dropout"] = nn.Dropout(input_dropout)

    n_layers = len(dims)

    if stochastic and n_layers == 1:
        encoder_layers["mu_logvar"] = IsotropicGaussianModule(n_items, dims[0])
        return nn.Sequential(encoder_layers)

    # n_layers > 1
    if stochastic:
        z_dim = dims[-1]
        # upto last layer
        dims = dims[:-1]

    # n_layers > 1
    prev_dim = n_items
    for i, next_dim in enumerate(dims):
        encoder_layers[f"layer_{i}"] = nn.Linear(in_features=prev_dim, out_features=next_dim)
        # every layer but the last layer has to have a non_linearity
        # this doesn't apply if stochastic is set, since the last layer is going to
        # be the gaussian layer
        if stochastic:
            encoder_layers[f"activation_{i}"] = non_linearity()
        elif i != len(dims) - 1:
            encoder_layers[f"activation_{i}"] = non_linearity()
        prev_dim = next_dim

    if stochastic:
        encoder_layers["mu_logvar"] = IsotropicGaussianModule(prev_dim, z_dim)

    return nn.Sequential(encoder_layers)


class IsotropicGaussianModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.mu(x), self.logvar(x)


def get_nonlinearity(name):
    if name.lower() == "relu":
        return nn.ReLU
    elif name.lower() == "tanh":
        return nn.Tanh


### LOSSES

def reconstruction_loss(x_prime, x):
    return -(F.log_softmax(x_prime, dim=-1) * x).sum(dim=-1).mean()


def kl(mu, logvar):
    return torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))


class AE(nn.Module):

    def __init__(self, n_items,
                 dims, input_dropout, non_linearity, ):
        super().__init__()
        self.n_items = n_items
        assert isinstance(dims, str), "`dims` should be a string"
        dims = [int(_) for _ in dims.split(",")]
        log.info(f"Dims: {dims}")
        self.dims = dims
        self.non_linearity = get_nonlinearity(non_linearity)
        self.input_dropout = input_dropout
        self.encoder = self.make_encoder()
        self.latent_dim = self.dims[-1]
        self.decoder = self.make_decoder()

    @abstractmethod
    def make_encoder(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def make_decoder(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, x, *args):
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, x, return_z=False, **kwargs):
        raise NotImplementedError()

    def decode(self, z):
        return self.decoder(z)

    def infer(self, x, return_z=False):
        z = self.encode(x)
        if return_z:
            return z, self.decode(z)
        return self.decode(z)

    def __str__(self):
        raise NotImplementedError()
