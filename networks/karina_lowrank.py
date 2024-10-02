# Copyright (c) 2024, Minjong Cheon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
import math
import os

def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_(mean, std).fmod_(2).mul_(std).add_(mean)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))

        # Initialize parameters
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        x = torch.matmul(x, self.B.t())
        return torch.matmul(x, self.A.t())

class GeoCyclicPadding(nn.Module):
    def __init__(self, pad_width):
        super(GeoCyclicPadding, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        circular_padded = torch.cat([x[:, :, :, -self.pad_width:], x, x[:, :, :, :self.pad_width]], dim=3)
        top_bottom_padded = torch.zeros(batch_size, channels, height + 2 * self.pad_width, circular_padded.shape[3], device=x.device)
        top_bottom_padded[:, :, self.pad_width:height + self.pad_width, :] = circular_padded
        middle_index = width // 2
        for i in range(self.pad_width):
            top_row = (self.pad_width - i-1) % height
            top_padding = torch.cat((circular_padded[:, :, top_row, middle_index:], circular_padded[:, :, top_row, :middle_index]), dim=-1)
            top_padding = top_padding.reshape(batch_size, channels, 1, -1)
            top_bottom_padded[:, :, i, :] = top_padding[:, :, 0, :]
            bottom_row = (height - i - 1) % height
            bottom_padding = torch.cat((circular_padded[:, :, bottom_row, middle_index:], circular_padded[:, :, bottom_row, :middle_index]), dim=-1)
            bottom_padding = bottom_padding.reshape(batch_size, channels, 1, -1)
            top_bottom_padded[:, :, height + self.pad_width + i, :] = bottom_padding[:, :, 0, :]
        return top_bottom_padded

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, rank_factor=4):
        super().__init__()
        self.pad = GeoCyclicPadding(pad_width=3)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim) 
        self.se = SELayer(dim)
        self.norm = LayerNorm(dim, eps=1e-6)

        rank = max(dim // rank_factor, 1)
        self.pwconv1 = LowRankLinear(dim, 4 * dim, rank) 
        self.act = nn.GELU()
        self.pwconv2 = LowRankLinear(4 * dim, dim, rank)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pad(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        x = input + self.drop_path(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LowRankConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, padding=0):
        super(LowRankConv2d, self).__init__()
        self.rank = rank
        self.conv1 = nn.Conv2d(in_channels, rank, (kernel_size, 1), padding=(padding, 0))
        self.conv2 = nn.Conv2d(rank, out_channels, (1, kernel_size), padding=(0, padding))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=67, num_classes=67, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            DepthwiseSeparableConv(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                DepthwiseSeparableConv(dims[i], dims[i+1], kernel_size=3, stride=1, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=1e-6) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[-1], num_classes, kernel_size=1)
        )

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.quant(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = self.dequant(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class KARINA(nn.Module):
    def __init__(self, img_size=(72, 144), in_chans=67, out_chans=67, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.core_model = ConvNeXt(in_chans=self.in_chans, num_classes=self.out_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate)

    def forward(self, x):
        x = self.core_model(x)
        return x

if __name__ == "__main__":
    model = KARINA()
    model.eval()
    
    # Fuse Conv, BatchNorm and ReLU modules (if applicable)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate the model with a few batches of training data
    sample = torch.randn(1, 67, 72, 144)  # Example input tensor
    model(sample)
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    
    # Check the model size
    def print_model_size(model):
        torch.save(model.state_dict(), "temp.p")
        os.remove("temp.p")