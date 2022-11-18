from src import arch


class MultDAE(arch.AE):

    def __init__(self, n_items,
                 dims, input_dropout, lam, non_linearity, **kwargs):
        super().__init__(n_items=n_items, dims=dims, input_dropout=input_dropout, non_linearity=non_linearity)
        self.lam = lam

    def make_encoder(self):
        return arch.make_encoder(self.n_items,
                                 self.dims,
                                 non_linearity=self.non_linearity,
                                 input_dropout=self.input_dropout,
                                 stochastic=False)

    def make_decoder(self):
        dec_dims = self.dims[::-1]
        dec_dims.append(self.n_items)
        return arch.make_decoder(
            dec_dims, dec_dims[0], self.non_linearity)

    def encode(self, x, *args):
        return self.encoder(x)

    def compute_loss(self, x, return_z=False, **kwargs):
        z, x_prime = self.infer(x, return_z=True)
        recon_loss = arch.reconstruction_loss(x_prime, x)

        ext = {
            "recon": recon_loss.item()
        }

        loss = recon_loss

        if return_z:
            return loss, ext, z
        return loss, ext

    def __str__(self):
        return f"MultDAE(dims={self.dims}, dropout={self.input_dropout}, lam={self.lam},\n  encoder={self.encoder},\ndecoder={self.decoder})"
