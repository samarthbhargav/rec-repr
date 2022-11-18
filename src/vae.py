import torch
from torch import nn

from src import arch
import logging

log = logging.getLogger(__name__)


class MultVAE(arch.AE):

    def __init__(self, n_items,
                 dims, input_dropout, non_linearity, anneal_cap,
                 total_anneal_steps, **kwargs):
        super().__init__(n_items=n_items, dims=dims, input_dropout=input_dropout,
                         non_linearity=non_linearity)
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps

    def make_encoder(self) -> nn.Module:
        return arch.make_encoder(self.n_items,
                                 self.dims,
                                 non_linearity=self.non_linearity,
                                 input_dropout=self.input_dropout,
                                 stochastic=True)

    def make_decoder(self) -> nn.Module:
        dec_dims = self.dims[::-1]
        dec_dims.append(self.n_items)
        return arch.make_decoder(
            dec_dims, dec_dims[0], self.non_linearity)

    def encode(self, x, *args):
        mu, logvar = self.mu_logvar(x)
        return self.reparameterize(mu, logvar)

    def mu_logvar(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def compute_loss(self, x, return_z=False, **kwargs):
        mu, logvar = self.mu_logvar(x)
        z = self.reparameterize(mu, logvar)

        x_prime = self.decode(z)
        kl = arch.kl(mu, logvar)
        recon_loss = arch.reconstruction_loss(x_prime, x)

        if self.total_anneal_steps > 0:
            train_step = kwargs["train_step"]
            anneal = min(self.anneal_cap, 1. * train_step / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        ext = {
            "recon": recon_loss.item(),
            "kl": kl.item(),
            "anneal_val": anneal
        }

        loss = recon_loss + (anneal * kl)

        if return_z:
            return loss, ext, z
        return loss, ext

    def __str__(self):
        return f"MultVAE(dims={self.dims}, dropout={self.input_dropout}, beta={self.anneal_cap} (steps={self.total_anneal_steps}),\n  encoder={self.encoder},\ndecoder={self.decoder})"
