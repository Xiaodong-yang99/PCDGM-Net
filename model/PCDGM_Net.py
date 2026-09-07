# *************************************************************************************************** #
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from functools import partial

# 安装mmcv参考链接：https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html#
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
# pip install torch_dct
import torch_dct as DCT
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from timm.models.layers import DropPath
from typing import Callable

# 参考文献[48]、[46]
'''
卷基层
'''
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

'''
BPC蓝图卷积
'''
class BPC(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1, dilation=1):
        super(BPC, self).__init__()
        self.main = nn.Sequential(
            conv_layer(in_channels, out_channels, 1),
            conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels)
        )

    def forward(self, x):
        return self.main(x)

'''
DSC深度可分离卷积
'''
class DSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSC, self).__init__()
        self.main = nn.Sequential(
            conv_layer(in_channels, out_channels, 3, groups=out_channels),
            conv_layer(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.main(x)

'''
Linear Layer的MLP
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # 恢复 [B, C, H, W]
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = (x1 + x2 + x3 + x4) / 4
    x_HL = (-x1 - x2 + x3 + x4) / 4
    x_LH = (-x1 + x2 - x3 + x4) / 4
    x_HH = (x1 - x2 - x3 + x4) / 4

    return x_LL, x_HL, x_LH, x_HH
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

def idwt_init(x_LL, x_HL, x_LH, x_HH):
    min_height = min(x_LL.size(2), x_HL.size(2), x_LH.size(2), x_HH.size(2))
    min_width = min(x_LL.size(3), x_HL.size(3), x_LH.size(3), x_HH.size(3))

    x_LL = x_LL[:, :, :min_height, :min_width]
    x_HL = x_HL[:, :, :min_height, :min_width]
    x_LH = x_LH[:, :, :min_height, :min_width]
    x_HH = x_HH[:, :, :min_height, :min_width]

    x1 = x_LL - x_HL - x_LH + x_HH
    x2 = x_LL - x_HL + x_LH - x_HH
    x3 = x_LL + x_HL - x_LH - x_HH
    x4 = x_LL + x_HL + x_LH + x_HH

    # 拼接回原始尺寸
    upper = torch.zeros(x1.size(0), x1.size(1), x1.size(2)*2, x1.size(3), device=x1.device, dtype=x1.dtype)
    lower = torch.zeros_like(upper)

    upper[:, :, 0::2, :] = x1
    upper[:, :, 1::2, :] = x2
    lower[:, :, 0::2, :] = x3
    lower[:, :, 1::2, :] = x4

    x = torch.zeros(x1.size(0), x1.size(1), upper.size(2), upper.size(3)*2, device=x1.device, dtype=x1.dtype)
    x[:, :, :, 0::2] = upper
    x[:, :, :, 1::2] = lower

    return x
class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        return idwt_init(x_LL, x_HL, x_LH, x_HH)


class DctSpatialInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct  # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
            )

        self.noise_threshold = nn.Parameter(torch.tensor(0.1))  # 控制高频抑制强度

    def forward(self, x):
        x, x_HL, x_LH, x_HH = dwt_init(x)
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)

        idct = torch.where(torch.abs(idct) > self.noise_threshold, torch.tanh(idct), idct)

        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')  # generate spatial mask
        out = idwt_init(x * dct_, x_HL * dct_, x_LH * dct_, x_HH * dct_)
        return out

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

class DctChannelInteraction(BaseModule):
    def __init__(self,
                 in_channels,
                 patch,
                 ratio,
                 isdct=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)],
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct:  # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.gelu(amaxp)) + self.channel1x1(self.gelu(aavgp))  # 2025 03 15 szc
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight  # filter out low-frequency features
        dct_ = DCT.idct_2d(dct, norm='ortho')

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.gelu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.gelu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

class HFP(BaseModule):
    def __init__(self,
                in_channels,
                ratio=(0.25, 0.25),
                patch = (8,8),
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(in_channels, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)

'''
EMA注意力
'''
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False, **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.spatial_scale = None
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.spatial_scale2 = None

    def forward(self, input):
        B, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()

        if self.spatial_scale is None:
            self.spatial_scale = nn.Parameter(torch.ones(H, W))
            self.register_parameter("spatial_scale", self.spatial_scale)

        if self.spatial_scale2 is None:
            self.spatial_scale2 = nn.Parameter(torch.ones(H, W))
            self.register_parameter("spatial_scale2", self.spatial_scale2)

        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                        1).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        spatial_scale_key = prefix + "spatial_scale"
        spatial_scale2_key = prefix + "spatial_scale2"

        if spatial_scale_key in state_dict and self.spatial_scale is None:
            scale_value = state_dict[spatial_scale_key]
            self.spatial_scale = nn.Parameter(scale_value)
            self.register_parameter("spatial_scale", self.spatial_scale)

        if spatial_scale2_key in state_dict and self.spatial_scale2 is None:
            scale2_value = state_dict[spatial_scale2_key]
            self.spatial_scale2 = nn.Parameter(scale2_value)
            self.register_parameter("spatial_scale2", self.spatial_scale2)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )



class LowFreqResidualCorrection(nn.Module):
    """低频残差矫正模块"""
    def __init__(self, dim):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.res_block = nn.Sequential(
            conv_layer(dim, dim, kernel_size=3, groups=dim),
            nn.GELU(),
            conv_layer(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        ll, hl, lh, hh = self.dwt(x)
        ll = ll + self.res_block(ll)
        out = self.idwt(ll, hl, lh, hh)
        return out



class HPENLiteSemanticBranch(nn.Module):
    """
    借鉴 HPEN 的 MFEB / Global 思想，但不引入 TAB。
    目标：
    1) 用多尺度局部卷积增强细节与局部上下文
    2) 再用轻量全局模块做语义整合
    3) 作为“感知分支”，不要去接管低频背景
    """
    def __init__(self, dim):
        super().__init__()

        c1 = dim // 2
        c2 = dim - c1

        self.pre = nn.Sequential(
            conv_layer(dim, dim, kernel_size=3),
            nn.GELU()
        )

        # HPEN风格的双分支局部增强
        self.dw_branch = nn.Sequential(
            conv_layer(c1, c1, kernel_size=3, groups=c1),
            nn.GELU()
        )

        self.dil_branch = nn.Sequential(
            conv_layer(c2, c2, kernel_size=3, dilation=2),
            nn.GELU()
        )

        # 聚合
        self.aggr = nn.Sequential(
            conv_layer(dim, dim, kernel_size=1),
            nn.GELU()
        )

        # 轻量全局细化：直接复用你现有的 VSSBlock
        self.context = VSSBlock(
            hidden_dim=dim,
            is_light_sr=False
        )

        # 末端细化
        self.out_proj = nn.Sequential(
            conv_layer(dim, dim, kernel_size=3),
            nn.GELU(),
            conv_layer(dim, dim, kernel_size=3)
        )

        # 语义分支的目标先验调制强度
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 残差注入强度
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, spatial_prior=None):
        feat = self.pre(x)
        x1, x2 = torch.split(feat, [feat.shape[1] // 2, feat.shape[1] - feat.shape[1] // 2], dim=1)

        b1 = self.dw_branch(x1)
        b2 = self.dil_branch(x2)

        feat = self.aggr(torch.cat([b1, b2], dim=1)) + x
        feat = self.context(feat)

        if spatial_prior is not None:
            feat = feat * (1.0 + torch.tanh(self.beta) * spatial_prior)

        feat = self.out_proj(feat)

    # 只返回语义残差，不返回 identity + residual
        residual = torch.tanh(self.gamma) * feat
        return residual


class PriorRestrictedChannelFusion(nn.Module):
    """
    借鉴 LDynSR 的动态加权思想，但只做通道级 gating。
    融合策略：
        out = x_struct + gate * prior * x_sem
    这样结构分支作为底座，语义分支只在目标相关区域注入。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_struct, x_sem, spatial_prior=None):
        feat = torch.cat([x_struct, x_sem], dim=1)
        gate = self.gate(self.pool(feat))   # [B, C, 1, 1]

        if spatial_prior is not None:
            if spatial_prior.shape[-2:] != x_struct.shape[-2:]:
                spatial_prior = F.interpolate(
                    spatial_prior,
                    size=x_struct.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            out = x_struct + gate * spatial_prior * x_sem
        else:
            out = x_struct + gate * x_sem

        return out
        
class STFRM(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)

        # local branch
        self.hfp = HFP(dim)
        self.act = nn.GELU()
        self.ema1 = EMA(dim)

        # target prior
        self.prior_gen = SpatialPriorGenerator(dim)

        # 结构全局分支：保持不动
        self.vss_struct = LowFreqTargetAwareVSSBlock(dim)
        self.ema2_struct = EMA(dim)
        self.lf_correction = LowFreqResidualCorrection(dim)

        # 语义全局分支：换成 HPEN-lite
        self.sem_branch = HPENLiteSemanticBranch(dim)
        self.ema2_sem = EMA(dim)

        # 融合：通道级动态 gate + prior 限制
        self.global_fusion = PriorRestrictedChannelFusion(dim)


        # 保留原 fusion_attn
        self.fusion_attn = nn.Sequential(
            conv_layer(2 * dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # -------------------------
        # local branch
        # -------------------------
        x1 = self.hfp(x)
        x1 = self.mlp(x1)
        x1 = self.act(x1)
        x1 = self.ema1(x1)

        # -------------------------
        # 生成目标先验
        # -------------------------
        spatial_prior = self.prior_gen(x1)

        # -------------------------
       # global branch: structure
       # -------------------------
        x2_struct = self.vss_struct(x, spatial_prior=spatial_prior)
        x2_struct = self.ema2_struct(x2_struct)
        x2_struct = self.lf_correction(x2_struct)

        # -------------------------
        # global branch: semantic
        # -------------------------
        x2_sem = self.sem_branch(x, spatial_prior=spatial_prior)
        x2_sem = self.ema2_sem(x2_sem)

       # -------------------------
       # prior-restricted channel fusion
       # -------------------------
        x2 = self.global_fusion(x2_struct, x2_sem, spatial_prior)

        # -------------------------
        # 融合到局部分支
        # -------------------------
        fusion_weight = self.fusion_attn(torch.cat([x1, x2], dim=1))
        out = x1 * fusion_weight + x2 * (1 - fusion_weight)

        return out


class CRFFM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.A1 = STFRM(dim)


    def forward(self, x):

        x = self.A1(x)  # x1 = A1(x1)

        return x

class DFE(nn.Module):
    def __init__(self, dim, num):
        super(DFE, self).__init__()

        self.crffm_blocks = nn.Sequential()
        for i in range(num):
            crffm = CRFFM(dim)
            self.crffm_blocks.add_module(f'crffm_{i}', crffm)

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        out = self.res_scale * self.crffm_blocks(x) + x
        return out

class OutBlock_lightweight(nn.Module):
    def __init__(self, in_channel, up_scale = 4):
        super(OutBlock_lightweight, self).__init__()
        self.up_scale = up_scale
        self.main = nn.Sequential(
            conv_layer(in_channel, 3*(self.up_scale**2), kernel_size=3),
            nn.PixelShuffle(self.up_scale),
            conv_layer(3, 3, kernel_size=3)
        )
    def forward(self, x):
        return self.main(x)

class OutBlock_lightweight2(nn.Module):
    def __init__(self, in_channel, up_scale = 4):
        super(OutBlock_lightweight2, self).__init__()
        self.up_scale = up_scale
        self.denoise_conv = nn.Sequential(
            conv_layer(in_channel, in_channel, kernel_size=3, groups=in_channel),  # 深度可分离卷积平滑
            nn.GELU()
        )
        if up_scale == 4:
            self.main = nn.Sequential(
                conv_layer(in_channel, in_channel*2, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//2, in_channel, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//4, 3, kernel_size=3)
            )
        else:
            self.main = nn.Sequential(
                conv_layer(in_channel, in_channel*2, kernel_size=3),
                nn.PixelShuffle(2),
                conv_layer(in_channel//2, 3, kernel_size=3)
            )
    def forward(self, x):
        x = self.denoise_conv(x)
        return self.main(x)

class SpatialPriorGenerator(nn.Module):
    """
    从局部分支特征中生成目标空间先验:
    输入:  [B, C, H, W]
    输出:  [B, 1, H, W]
    """
    def __init__(self, dim):
        super(SpatialPriorGenerator, self).__init__()
        hidden = max(dim // 8, 4)
        self.prior = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.prior(x)


class LowFreqTargetAwareVSSBlock(nn.Module):
    """
    低频结构引导的目标感知全局分支
    1) 先用 DWT 提取低频 LL
    2) 上采样回原尺寸，作为全局结构输入
    3) 用局部分支产生的 spatial prior 做弱调制
    4) 送入 VSSBlock 建模
    5) 用残差式安全注入返回，避免一开始训练不稳定
    """
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.dwt = DWT()

        # 低频预处理
        self.low_proj = nn.Sequential(
            conv_layer(hidden_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            conv_layer(hidden_dim, hidden_dim, kernel_size=3)
        )

        # 原始 VSS 作为核心全局建模器
        self.vss = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            expand=expand,
            is_light_sr=is_light_sr,
            **kwargs,
        )

        # 输出细化
        self.out_proj = nn.Sequential(
            conv_layer(hidden_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            conv_layer(hidden_dim, hidden_dim, kernel_size=3)
        )

        # 先验调制强度，初始化为 0，保证训练初期接近 baseline
        self.alpha_in = nn.Parameter(torch.tensor(0.0))
        self.alpha_out = nn.Parameter(torch.tensor(0.0))

        # 残差注入强度，初始化为 0，训练更稳
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, spatial_prior=None):
        """
        x: [B, C, H, W]
        spatial_prior: [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # 1) 提取低频 LL
        x_ll, _, _, _ = self.dwt(x)

        # 2) 恢复到原始分辨率
        x_ll = F.interpolate(x_ll, size=(H, W), mode='bilinear', align_corners=False)
        x_ll = self.low_proj(x_ll)

        # 3) 用目标先验做弱调制
        if spatial_prior is not None:
            x_ll = x_ll * (1.0 + torch.tanh(self.alpha_in) * spatial_prior)

        # 4) 全局建模
        global_feat = self.vss(x_ll)

        # 5) 输出端再做一次轻度目标强化
        if spatial_prior is not None:
            global_feat = global_feat * (1.0 + torch.tanh(self.alpha_out) * spatial_prior)

        global_feat = self.out_proj(global_feat)

        # 6) 残差式安全注入
        out = x + torch.tanh(self.gamma) * global_feat
        return out

class MYMODEL(nn.Module):
    def __init__(self, up_scale=2):
        super(MYMODEL, self).__init__()
        self.up_scale = up_scale
        self.basechannles = 64
        self.modulenum = 4

        self.sfe = conv_layer(3, self.basechannles, kernel_size=3)
        self.dfe = DFE(self.basechannles,self.modulenum)
        self.rec = OutBlock_lightweight2(self.basechannles, self.up_scale)

    def forward(self, input):
        out = self.sfe(input)
        out = self.dfe(out)
        out = self.rec(out)
        return out