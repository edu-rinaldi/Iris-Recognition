import os
from collections import OrderedDict
from torch import nn, load

class FeatNet(nn.Module):
    def __init__(self, pretrainedName=""):
        super(FeatNet, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_a', nn.Conv2d(3, 16, kernel_size=(3, 7), stride=1, padding=(1, 3), bias=False)),
            ('tan1_a', nn.Tanh())
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('pool1_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2_a', nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False)),
            ('tan2_a', nn.Tanh())
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('pool2_a', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv3_a', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('tan3_a', nn.Tanh())
        ]))
        self.fuse_a = nn.Conv2d(112, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten()

        if pretrainedName != '':
            modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pretrained", pretrainedName)
            self.load_state_dict(load(modelPath))
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2 = F.interpolate(x2, size=(64, 200), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(64, 200), mode='bilinear', align_corners=False)
        x4 = torch.cat((x1, x2, x3), dim=1)
        out = self.fuse_a(x4)
        out = self.flatten(out)
        return out