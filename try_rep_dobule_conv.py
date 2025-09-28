import torch
from torch import nn

from networks.models.UNet import double_conv
from networks.models.convnext import Block

mpconv = nn.Sequential(
    nn.MaxPool2d(2),
    double_conv(64, 128,group_conv=False)
)

convnext_conv = nn.Sequential(
    nn.Conv2d(64,128, kernel_size=2, stride=2),
    Block(128)
)

if __name__ == "__main__":
    img = torch.rand(1,64,512,512).cuda()
    mp = mpconv.cuda()
    conv = convnext_conv.cuda()
    print(mp(img).shape)
    print(conv(img).shape)