import torch
import torch.nn as nn

Pool = nn.MaxPool2d

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)

        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)

        return up1 + up2


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, net,  joint_num, inp_dim=256, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        nstack = int(net.split('_')[-1])
        self.nstack = nstack
        self.joint_num = joint_num
        self.pre = nn.Sequential(
            Conv(1, 64, 5, 1, bn=True, relu=True),
            # Conv(1, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 256),
            Residual(256, inp_dim)
        ) # keep downsample rate to 2
        # self.pre = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm2d(64, momentum=0.1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs_1 = nn.ModuleList([nn.Conv2d(in_channels=inp_dim, out_channels=self.joint_num * 3, kernel_size=1, stride=1, padding=0) for i in range(nstack)])
        self.outs_2 = nn.ModuleList([nn.Conv2d(in_channels=inp_dim, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0) for i in range(nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(self.joint_num * 4, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack

    def forward(self, imgs):
        ## our posenet
        x = self.pre(imgs)
        # print(x.shape)
        combined_hm_preds = []
        combined_feature = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            offset = self.outs_1[i](feature)
            # print(offset.shape)
            dis = self.outs_2[i](feature)
            # print(dis.shape)
            preds = torch.cat((offset, dis), dim=1)
            # preds = torch.cat((offset, dis), dim=1)
            combined_hm_preds.append(preds)
            combined_feature.append(feature)

            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return combined_hm_preds

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # img = torch.rand([1, 1, 128, 128]).cuda()
    # print(img.size())
    net = PoseNet('hourglass_1', 14).cuda()
    # out = net(img)
    # print(len(out))
    # print(out[-1].size())
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (1, 128, 128), as_strings=False, print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs*2))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
