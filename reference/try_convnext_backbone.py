# from networks.models.convnext_for_completion import ConvNeXt
import torch
# model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 512],in_chans=64).cuda()
#
# down1_in = torch.randn(1,64,256,256).cuda()
# down2_in = torch.randn(1,256,128,128).cuda()
# down3_in = torch.randn(1,512,64,64).cuda()
# down4_in = torch.randn(1,1024,32,32).cuda()
#
#
# # downsample_layers[i](x)
# # x = self.stages[i](x)
# x2 = model.downsample_layers[0](down1_in)
# x2 = model.stages[0](x2)
#
# x3 = model.downsample_layers[1](down2_in)
# x3 = model.stages[1](x3)
#
# x4 = model.downsample_layers[2](down3_in)
# x4 = model.stages[2](x4)
#
# x5 = model.downsample_layers[3](down4_in)
# x5 = model.stages[3](x5)
#
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)
# print(x5.shape)

input = torch.randn(1,3,256,256).cuda(1)
model = torch.nn.Conv2d(3,64,kernel_size=3,padding=1).cuda(1)
output = model(input)
pass