# all models from https://arxiv.org/pdf/1512.03385.pdf

import torch.nn as nn
from functools import partial


__all__ = ['ResNet', 'ResNetBackbone', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet18_bb', 'resnet34_bb', 'resnet50_bb', 'resnet101_bb', 'resnet152_bb']


def activation_func(activation):
    """ a dictionary with different activation functions"""
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def conv_with_bn(in_channels, out_channels, conv, *args, **kwargs):
    """ stack one convolution and batchnormalization layer """
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    """
    Main building block:
        ResidualBlock(
            (blocks): Identity()
            (activate): ReLU(inplace)
            (shortcut): Identity()
            )
    """

    def __init__(self, in_channels, out_channels, activation='relu'):
        """
        :param activation: type of activation function
        """
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    """Just extend residual block and define the shortcut function"""

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    """Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation"""
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_with_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_with_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    """
    The three layers are 1x1, 3x3, and 1x1 convolutions to increase the network depth
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_with_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_with_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_with_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    The tail of ResNet
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    """ Put all the pieces together """
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNetBackbone(nn.Module):
    """ Encoder """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)

    def forward(self, x):
        return self.encoder(x)


def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    """
    ResNet-18
    Total params: 11,689,512
    Trainable params: 11,689,512
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 65.86
    Params size (MB): 44.59
    Estimated Total Size (MB): 111.03
    ----------------------------------------------------------------

    :param in_channels: input channels
    :param n_classes: number of output classes
    :param block: type of basic block of net
    :return: instance of net
    """
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)


def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    """
    ResNet-34
    Total params: 21,797,672
    Trainable params: 21,797,672
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 99.36
    Params size (MB): 83.15
    Estimated Total Size (MB): 183.08
    ----------------------------------------------------------------

    :param in_channels: input channels
    :param n_classes: number of output classes
    :param block: type of basic block of net
    :return: instance of net
    """
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet50(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    """
    ResNet-50
    Total params: 25,557,032
    Trainable params: 25,557,032
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 298.82
    Params size (MB): 97.49
    Estimated Total Size (MB): 396.88
    ----------------------------------------------------------------

    :param in_channels: input channels
    :param n_classes: number of output classes
    :param block: type of basic block of net
    :return: instance of net
    """
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet101(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    """
    ResNet-101
    Total params: 44,549,160
    Trainable params: 44,549,160
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 441.99
    Params size (MB): 169.94
    Estimated Total Size (MB): 612.50
    ----------------------------------------------------------------

    :param in_channels: input channels
    :param n_classes: number of output classes
    :param block: type of basic block of net
    :return: instance of net
    """
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)


def resnet152(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    """
    ResNet-152
    Total params: 60,192,808
    Trainable params: 60,192,808
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 618.85
    Params size (MB): 229.62
    Estimated Total Size (MB): 849.04
    ----------------------------------------------------------------

    :param in_channels: input channels
    :param n_classes: number of output classes
    :param block: type of basic block of net
    :return: instance of net
    """
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)


def resnet18_bb(in_channels, block=ResNetBasicBlock, *args, **kwargs):
    """ backbone of resnet18"""
    return ResNetBackbone(in_channels, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)


def resnet34_bb(in_channels, block=ResNetBasicBlock, *args, **kwargs):
    """ backbone of resnet34"""
    return ResNetBackbone(in_channels, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet50_bb(in_channels, block=ResNetBasicBlock, *args, **kwargs):
    """ backbone of resnet50"""
    return ResNetBackbone(in_channels, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)


def resnet101_bb(in_channels, block=ResNetBasicBlock, *args, **kwargs):
    """ backbone of resnet101"""
    return ResNetBackbone(in_channels, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)


def resnet152_bb(in_channels, block=ResNetBasicBlock, *args, **kwargs):
    """ backbone of resnet152"""
    return ResNetBackbone(in_channels, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)
