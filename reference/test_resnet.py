import torch
from torchvision import models

from reference.convnext import convnext_tiny_for_cm, convnext_small_for_cm
from resnet_for_mutil_cross_attention import resnet34, resnet18

if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512).cuda()
    model = resnet34().cuda()
    x4 = model(x)
    # print('resnet:', x1.shape)  # torch.Size([2, 64, 128, 128])
    # print('resnet:', x2.shape) # torch.Size([2, 128, 64, 64])
    # print('resnet:', x3.shape)  # torch.Size([2, 256, 32, 32])
    print('resnet:', x4.shape)  # torch.Size([2, 512, 16, 16])

    # # ================= test convnext =================
    # model = models.vgg16().cuda()
    # out = model(x)
    # print(out.shape)

    # ================= test SSA-SC =================
    # occu = torch.randn(1, 32, 256, 256)
    # out_data = torch.randn(1, 32, 256, 256)
    # x1 = torch.cat((occu, out_data), dim=1)  # 不cat
    #
    # x2_cat = torch.randn(1, 256, 128, 128)
    # x3_cat = torch.randn(1, 512, 64, 64)
    # x4_cat = torch.randn(1, 1024, 32, 32)
    # x5 = torch.randn(1, 512, 16, 16)  # 不cat

    # x = torch.randn(2, 3, 512, 512).cuda()
    # model = resnet34().cuda()
    # out = model(x)
    # out = out.contiguous().view((out.shape[0], out.shape[1], -1))
    # print(out.shape)
    # out = out.permute(0, 2, 1)
    # print(out.shape)
    # out = out.permute(0,2,1)
    # print(out.shape)
    # pass    https://blog.csdn.net/m0_61899108/article/details/125564706

