import math
import torch
import torch.nn as nn


BN_MOMENTUM = 0.1

def get_deconv_net(layers, num_classes, downsample):
    RESNET = {18: (BasicBlock, [2, 2, 2, 2]),
                    50: (Bottleneck, [3, 4, 6, 3]),
                    101: (Bottleneck, [3, 4, 23, 3]),
                    152: (Bottleneck, [3, 8, 36, 3])
                }
    block, layers = RESNET[layers]
    model = ResnetDeconv(1, num_classes, block, layers, downsample=downsample)
    return model


class ResnetDeconv(nn.Module):
    def __init__(self, inchannels, outchannels, block, layers=[2, 2, 2, 2], downsample=1):
        '''
        :params inchannels: input image channels, for depth image, inplanes=1
        :params outchannels: output channels, aks joint nums
        :params block: basic building block of resnet
        :params layers: num of blocks in each component of resnet
        :params downsample: downsample rate, indicating the downsample rate from input to output, i.e., input size 128 and output size 64, downsample=2

        '''
        super(ResnetDeconv, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(inchannels, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        deconv_num = 4 - int(math.log(downsample, 2))
        deconv_kernel = [4] * deconv_num
        deconv_planes = [256] * deconv_num
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(deconv_num, deconv_kernel, deconv_planes)

        # self.final1 = nn.Conv2d(in_channels=256, out_channels=outchannels*2, kernel_size=1, stride=1, padding=0)
        self.final1 = nn.Conv2d(in_channels=256, out_channels=outchannels*3, kernel_size=1, stride=1, padding=0)
        self.final2 = nn.Conv2d(in_channels=256, out_channels=outchannels, kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, kernels, planes):

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = kernels[i], 1, 0
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes[i],
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes[i], momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes[i]

        return nn.Sequential(*layers)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

        for m in self.final1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.final2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        c = self.pre(x)
        # print('pre size: ', x.size())

        c1 = self.layer1(c)
        # print('enc1 size: ', c1.size())
        c2 = self.layer2(c1)
        # print('enc2 size: ', c2.size())
        c3 = self.layer3(c2)
        # print('enc3 size: ', c3.size())
        c4 = self.layer4(c3)
        # print('enc4 size: ', c4.size())

        out = self.deconv_layers(c4)
        # print('deconv size: ', x.size())
        vec = self.final1(out)
        ht = self.final2(out)

        return torch.cat([vec, ht], dim=1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    img = torch.randn(1, 1, 128, 128).cuda()
    model = get_deconv_net(18, 14, 2).cuda()
    print(model)
    out = model(img)
    print(out.size())
    macs, params = get_model_complexity_info(model, (1,128,128),as_strings=True,print_per_layer_stat=False)
    print(macs)
    print(params)
