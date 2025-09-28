import torch
import torch.nn as nn
from networks.models.CBAM import CBAMBlock


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(CBAMBlock(in_channels), nn.BatchNorm2d(in_channels))
        self.act = nn.SiLU(True)

    def forward(self, x):
        out = self.attention(x)
        out += x
        out = self.act(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()
        self.attention = nn.Sequential(CBAMBlock(in_channels), nn.BatchNorm2d(in_channels))
        self.act = nn.SiLU(True)
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                    nn.SiLU(True))

    def forward(self, pc, im):
        input = torch.cat([pc, im], dim=1)
        x = self.attention(input)
        x += input
        x = self.act(x)
        out = self.reduce(x)
        return out


class CrossModalFusion(nn.Module):
    def __init__(self, in_channels=512):
        super(CrossModalFusion, self).__init__()


        self.cross_atten_1 = CrossAttention(in_channels * 2, in_channels)

        self.self_attn_1 = SelfAttention(in_channels)

        self.cross_atten_2 = CrossAttention(in_channels * 2, in_channels)

        self.self_attn_2 = SelfAttention(in_channels)

        self.cross_atten_3 = CrossAttention(in_channels * 2, in_channels)

    def forward(self, pc, im):
        pc = self.cross_atten_1(pc, im)

        pc = self.self_attn_1(pc)

        pc_skip = pc

        pc = self.cross_atten_2(pc, im)

        pc = self.self_attn_2(pc)

        pc = self.cross_atten_3(pc, pc_skip)

        return pc
