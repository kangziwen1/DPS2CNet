import torch
import torch.nn as nn


class AggreAttention(nn.Module):
    def __init__(self, dims, heads=4):
        super(AggreAttention, self).__init__()
        self.dims = dims
        self.heads = heads
        self.cross_atten = nn.MultiheadAttention(dims, heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(dims)

        self.self_attn = nn.MultiheadAttention(dims, heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(dims)

    def forward(self, pc_q, im_k):
        hw = im_k.shape[-1]
        im_k = im_k.contiguous().view((im_k.shape[0], im_k.shape[1], -1))  # 2 512 HW
        # im_k = im_k.permute(0, 2, 1)

        im_v = im_k
        pc_q = pc_q.contiguous().view((pc_q.shape[0], pc_q.shape[1], -1))
        # pc_q = pc_q.permute(0, 2, 1)

        x, _ = self.cross_atten(pc_q, im_k, im_v)
        pc_q = self.layer_norm_1(x + pc_q)

        x, _ = self.self_attn(pc_q, pc_q, pc_q)
        pc_q = self.layer_norm_2(x + pc_q)
