from torch import nn


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


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=2, padding=1):
        super().__init__()
        self.resLayers = []
        for i in depth:
            self.resLayers.appemd(ResidualConnection(in_dim, in_dim, kernel, stride, padding))
        self.down = DownSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.resLayers:
            x = layer(x)
        x = self.down(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer_dims=[16, 32, 64, 128], input_dim=128, layer_depth=2):
        super().__init__()

        self.Blocks = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.appemd(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))

    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layer_dims=[128, 64, 32, 16], output_dim=128, layer_depth=2):
        super().__init__()

        self.Blocks = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.appemd(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))

    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
        return x
