import torch
import torch.nn as nn
# TODO https://blog.csdn.net/amusi1994/article/details/122335513

class CrossModalFusion(nn.Module):
    def __init__(self, dims, heads):
        super(CrossModalFusion, self).__init__()
        # 1, 512, 16, 16
        self.dims = dims
        self.heads = heads
        self.dropout = 0.5
        # self.img_channel_Alignment = nn.Conv2d(256, 512, 1)

        self.cross_atten_1 = nn.MultiheadAttention(dims, heads, batch_first=True, dropout=self.dropout)
        self.layer_norm_1 = nn.LayerNorm(dims)
        self.act_1 = nn.SiLU(True)

        self.self_attn_1 = nn.MultiheadAttention(dims, heads, batch_first=True, dropout=self.dropout)
        self.layer_norm_2 = nn.LayerNorm(dims)
        self.act_2 = nn.SiLU(True)

        self.cross_atten_2 = nn.MultiheadAttention(dims, heads, batch_first=True, dropout=self.dropout)
        self.layer_norm_3 = nn.LayerNorm(dims)
        self.act_3 = nn.SiLU(True)

        self.self_attn_2 = nn.MultiheadAttention(dims, heads, batch_first=True, dropout=self.dropout)
        self.layer_norm_4 = nn.LayerNorm(dims)
        self.act_4 = nn.SiLU(True)

        self.cross_atten_3 = nn.MultiheadAttention(dims, heads, batch_first=True, dropout=self.dropout)
        self.layer_norm_5 = nn.LayerNorm(dims)
        self.act_5 = nn.SiLU(True)

    def forward(self, pc_q, im_k):
        # pc_q torch.Size([2, 512, 16, 16])

        hw = im_k.shape[-1]
        im_k = im_k.contiguous().view((im_k.shape[0], im_k.shape[1], -1))  # 2 512 HW
        # im_k = im_k.permute(0, 2, 1)

        im_v = im_k
        pc_q = pc_q.contiguous().view((pc_q.shape[0], pc_q.shape[1], -1))
        # pc_q = pc_q.permute(0, 2, 1)

        x, _ = self.cross_atten_1(pc_q, im_k, im_v)
        pc_q = self.layer_norm_1(x + pc_q)
        pc_q = self.act_1(pc_q)

        x, _ = self.self_attn_1(pc_q, pc_q, pc_q)
        pc_q = self.layer_norm_2(x + pc_q)
        pc_q = self.act_2(pc_q)
        pc_skip = pc_q

        x, _ = self.cross_atten_2(pc_q, im_k, im_v)
        pc_q = self.layer_norm_3(x + pc_q)
        pc_q = self.act_3(pc_q)

        x, _ = self.self_attn_2(pc_q, pc_q, pc_q)
        pc_q = self.layer_norm_4(x + pc_q)
        pc_q = self.act_4(pc_q)

        x, _ = self.cross_atten_3(pc_q, pc_skip, pc_skip)
        pc_q = self.layer_norm_5(x + pc_q)
        pc_q = self.act_5(pc_q)
        # pc_q = pc_q.permute(0, 2, 1)
        pc_q = pc_q.contiguous().view((pc_q.shape[0], pc_q.shape[1], hw, hw))

        return pc_q
