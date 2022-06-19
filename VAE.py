from torch import nn
from torch import exp, randn_like


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyRelu()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
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
    def __init__(self, in_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        self.resLayer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel, stride, padding),
            nn.BatchNorm2d(in_dim),
            nn.LeakyRelu()
        )

    def forward(self, x):
        residual = x
        x = self.resLayer(x)
        x = self.resLayer(x)
        x = nn.add(residual, x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=2, padding=1):
        super().__init__()
        self.upLayers = []
        for i in depth:
            self.resLayers.append(ResidualConnection(in_dim, in_dim, kernel, stride, padding))
        self.down = DownSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.resLayers:
            x = layer(x)
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=2, padding=1):
        super().__init__()
        self.resLayers = []
        for i in depth:
            self.resLayers.append(ResidualConnection(in_dim, in_dim, kernel, stride, padding))
        self.down = DownSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.resLayers:
            x = layer(x)
        x = self.down(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer_dims=[3, 8, 16, 32, 64, 128, 256], input_size=128, latent_size = 32, layer_depth=2):
        super().__init__()

        self.Blocks = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.append(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))
        out_size = input_size/((2**len(layer_dims)-1))
        self.out_mu = nn.Linear((out_dim, out_size, out_size), latent_size)
        self.out_var = nn.Linear((out_dim, out_size, out_size), latent_size)


    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
            mu = self.out_mu
            var = self.out_var
        return [mu, var]


class Decoder(nn.Module):
    def __init__(self, layer_dims=[256, 128, 64, 32, 16, 8, 3], output_size=128, latent_size = 32, layer_depth=2):
        super().__init__()
        in_dim = layer_dims[0]
        in_size = output_size/((2**len(layer_dims)-1))
        self.up_size = nn.Linear( latent_size, (in_dim, in_size, in_size))

        self.Blocks = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.append(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))

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
        x = self.reparameterize(mu, var)
        for block in self.Blocks:
            x = block(x)
        return x


