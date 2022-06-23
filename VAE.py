import torch
from torch import nn,  exp, randn_like
from torch.nn import functional as F

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.subPixel = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel, stride, padding),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.subPixel(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, in_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.resLayer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel, stride, padding),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        residual = x
        x = self.resLayer(x)
        x = self.resLayer(x)
        x = torch.add(residual, x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=1, padding=1):
        super().__init__()
        self.upLayers = nn.ModuleList([])
        for i in range(depth):
            self.upLayers.append(ResidualConnection(in_dim, kernel, stride=1, padding=padding))
        self.up = UpSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.upLayers:
            x = layer(x)
        x = self.up(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=2, padding=1):
        super().__init__()
        self.resLayers = nn.ModuleList([])
        for i in range(depth):
            self.resLayers.append(ResidualConnection(in_dim, kernel, stride=1, padding=padding))
        self.down = DownSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.resLayers:
            x = layer(x)
        x = self.down(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer_dims=[3, 8, 16, 32, 64, 128, 256], input_size=128, latent_size = 32, layer_depth=2):
        super().__init__()

        self.Blocks = nn.ModuleList([])

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.append(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))
        self.out_dim = layer_dims[-1]
        self.out_size = input_size//(2**(len(layer_dims)-1))
        self.out_mu = nn.Linear(self.out_dim * self.out_size * self.out_size, latent_size)
        self.out_var = nn.Linear(self.out_dim * self.out_size * self.out_size, latent_size)


    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
        mu = self.out_mu(x.reshape(-1, self.out_dim * self.out_size * self.out_size))
        var = self.out_var(x.reshape(-1, self.out_dim * self.out_size * self.out_size))
        return mu, var


class Decoder(nn.Module):
    def __init__(self, layer_dims=[256, 128, 64, 32, 16, 8, 3], output_size=128, latent_size = 32, layer_depth=2):
        super().__init__()
        self.latent_size = latent_size
        self.in_dim = layer_dims[0]
        self.in_size = output_size//(2**(len(layer_dims)-1))
        self.up_size = nn.Linear(self.latent_size,  self.in_dim * self.in_size * self.in_size)

        self.Blocks =  nn.ModuleList([])

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.append(UpBlock(in_dim, out_dim, layer_depth, kernel=3, stride=1, padding=1))

    def reparameterize(self, mu, var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = exp(0.5 * var)
        eps = randn_like(std)
        return eps * std + mu

    def forward(self, mu, var):

        z = self.reparameterize(mu, var)
        x = self.up_size(z)
        x = x.reshape(-1, self.in_dim, self.in_size, self.in_size)
        for block in self.Blocks:
            x = block(x)


        return x

class VAE(nn.Module):
    def __init__(self, layer_dims=[3, 8, 16, 32, 64, 128, 256], data_size=128, latent_size=32, enc_layer_depth=2, dec_layer_depth=2):
        super().__init__()
        self.enc_layer_dims=layer_dims
        self.dec_layer_dim=list(reversed(layer_dims))
        self.data_size=data_size
        self.latent_size=latent_size
        self.enc_layer_depth=enc_layer_depth
        self.dec_layer_depth=dec_layer_depth

        self.encoder = Encoder(self.enc_layer_dims, self.data_size, self.latent_size, self.enc_layer_depth)
        self.decoder = Decoder(self.dec_layer_dim, self.data_size, self.latent_size, self.dec_layer_depth)

    def forward(self, x):
        mu, var = self.encoder(x)
        out = self.decoder(mu, var)

        return out, mu, var

    def decode(self, mu, var):
        out = self.decoder(mu, var)

        return out

    def encode(self, x):
        mu, var = self.encoder(x)

        return mu, var

    def calc_loss(self, _in, _out, mu, var, alpha = 1):

        mse_loss = F.mse_loss(_out, _in)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)

        #kld_weight = batch_size

        loss = alpha * mse_loss + kld_loss
        return {'loss': loss, 'MSE_Loss': mse_loss.detach(), 'KLD':-kld_loss.detach()}



