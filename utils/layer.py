import torch
import torch.nn as nn
# from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class LinearLayer(nn.Module):
    def __init__(self, feat_dims, relu_final=True, use_bn=False):
        super(LinearLayer, self).__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

            # Do not use ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
                if use_bn:
                    layers.append(nn.BatchNorm1d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        self.linear = nn.ModuleList(layers)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class ConvLayer(nn.Module):
    def __init__(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        super(ConvLayer, self).__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
                nn.Conv2d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                    ))
            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.conv = nn.ModuleList(layers)
    
    def forward(self, x):
        out = self.conv(x)
        return out

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class Conv1dLayer(nn.Module):
    def __init__(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        super(Conv1dLayer, self).__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
                nn.Conv1d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                    ))
            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.conv1d = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.conv1d:
            x = layer(x)
        return x


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    # return nn.Sequential(*layers)
    return nn.ModuleList(layers)

class DeconvLayer(nn.Module):
    def __init__(self, feat_dims, bnrelu_final=True):
        super(DeconvLayer, self).__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))

            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.deconv:
            x = layer(x)
        return x


if __name__ == "__main__":
    deconv = make_deconv_layers([2048,256,256,256])
    input = torch.randn((2, 2048, 8, 8))
    output = deconv(input)
    print(output.shape)

