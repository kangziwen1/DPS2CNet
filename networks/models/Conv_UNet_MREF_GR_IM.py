import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
# import torchsnooper
import torch_scatter
from CrossModalCompletion import CrossModalFusion
from networks.common.batchinstancenorm import BatchInstanceNorm2d
from networks.models.CBAM import CBAMBlock
from networks.models.aggre_attention import AggreAttention
from networks.models.convnext_for_completion import ConvNeXt_FC, LayerNorm


class down_convnext:
    def __init__(self, model, count, norm):
        self.model = model
        self.index = count - 1
        self.norm = norm

    def __call__(self, x):
        x = self.model.downsample_layers[self.index](x)
        x = self.model.stages[self.index](x)
        return self.norm(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(CBAMBlock(in_channels), BatchInstanceNorm2d(in_channels))
        self.act = nn.SiLU(True)

    def forward(self, x):
        out = self.attention(x)
        out += x
        out = self.act(out)
        return out


class BEV_UNet_Convnext_GR_IM(nn.Module):
    def __init__(self, n_class, n_height, dilation, bilinear, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock):
        super(BEV_UNet_Convnext_GR_IM, self).__init__()
        self.inc = inconv(64, 64, dilation, input_batch_norm, circular_padding)
        self.encoder = ConvNeXt_FC(depths=[3, 3, 27, 3], dims=[128, 256, 512, 512], in_chans=64).cuda()
        self.down1 = down_convnext(self.encoder, 1, BatchInstanceNorm2d(128).cuda())
        self.down2 = down_convnext(self.encoder, 2, BatchInstanceNorm2d(256).cuda())
        self.down3 = down_convnext(self.encoder, 3, BatchInstanceNorm2d(512).cuda())
        self.down4 = down_convnext(self.encoder, 4, BatchInstanceNorm2d(512).cuda())

        self.up1 = up(1536, 512, 512, circular_padding, bilinear=bilinear, group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout, cbam_num=0)
        self.up2 = up(1024, 256, 512, circular_padding, bilinear=bilinear, group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout, cbam_num=1)
        self.up3 = up(512, 128, 256, circular_padding, bilinear=bilinear, group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout, cbam_num=2)
        self.up4 = up(192, 128, 128, circular_padding, bilinear=bilinear, group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout, cbam_num=3)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(128, n_class)
        self.cross_model = CrossModalFusion(256, 4)

    def forward(self, x, x_ds1, x_ds2, x_ds3,im_encoder):
        x1 = self.inc(x)  # [B, 64, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 128, 128]
        x2_cat = torch.cat((x2, self.channel_reduction(x_ds1, x2.shape[1])),
                           dim=1)  # [B, 128, 128, 128] + [B, 128, 128, 128]    256
        x3 = self.down2(x2_cat)  # [B, 256, 64, 64]
        x3_cat = torch.cat((x3, self.channel_reduction(x_ds2, x3.shape[1])),
                           dim=1)  # [B, 256, 64, 64] + [B, 256, 64, 64]    512
        x4 = self.down3(x3_cat)  # [B, 512, 32, 32]
        x4_cat = torch.cat((x4, x_ds3), dim=1)  # [B, 512, 32, 32] + [B, 512, 32, 32]    1024
        x5 = self.down4(x4_cat)  # [B, 512, 16, 16]

        x5 = self.cross_model(x5, im_encoder)

        x = self.up1(x5, x4_cat)
        x = self.up2(x, x3_cat)
        x = self.up3(x, x2_cat)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: (B, C1, H, W)
            out_channels: C2

        Returns:

        """
        B, in_channels, H, W = x.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.view(B, out_channels, -1, H, W)
        # x = torch.max(x, dim=2)[0]
        x = x.sum(dim=2)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_bin(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_bin, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                BatchInstanceNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                BatchInstanceNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                BatchInstanceNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                BatchInstanceNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    BatchInstanceNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
            else:
                self.conv = nn.Sequential(
                    BatchInstanceNorm2d(in_ch),
                    double_conv_bin(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv_bin(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, up_ch, circular_padding, pix=False, bilinear=False, group_conv=False,
                 use_dropblock=False,
                 drop_p=0.5, cbam_num=1):
        super(up, self).__init__()
        self.cbam_num = cbam_num
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(up_ch, up_ch, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv_bin(in_ch, out_ch, group_conv=group_conv)
        # https://blog.csdn.net/e01528/article/details/84397174?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167093233516782395363556%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167093233516782395363556&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-84397174-null-null.142^v68^control,201^v4^add_ask,213^v2^t3_esquery_v2&utm_term=nn.modulelist&spm=1018.2226.3001.4187
        self.attention_gate_block = nn.ModuleList()

        for i in range(self.cbam_num):
            self.attention_gate_block.append(SelfAttention(out_ch))

        self.act = nn.Sequential(BatchInstanceNorm2d(up_ch), nn.SiLU(True))

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.act(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        for i in range(self.cbam_num):
            x = self.attention_gate_block[i](x)

        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return t
