'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, act):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            act,
            nn.Dropout(),
            nn.Linear(512, 512),
            act,
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, feature=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if feature:
            return x
        x = self.classifier(x)
        return x


def make_layers(cfg, act, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(activation, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], activation), activation)


def vgg11_bn(activation, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], activation, batch_norm=True), activation)


def vgg13(activation, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], activation), activation)


def vgg13_bn(activation, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], activation, batch_norm=True), activation)


def vgg16(activation, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], activation), activation)


def vgg16_bn(activation, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], activation, batch_norm=True), activation)


def vgg19(activation, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], activation), activation)


def vgg19_bn(activation, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], activation, batch_norm=True), activation)
