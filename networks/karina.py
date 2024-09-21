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

import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.registry import register_model

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

class GeoCyclicPadding(nn.Module):
    def __init__(self, pad_width):
        super(GeoCyclicPadding, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Circular padding for left and right
        circular_padded = torch.cat([x[:, :, :, -self.pad_width:], x, x[:, :, :, :self.pad_width]], dim=3)

        # Zero-padding tensor for top and bottom padding
        top_bottom_padded = torch.zeros(batch_size, channels, height + 2 * self.pad_width, circular_padded.shape[3], device=x.device)

        # Placing the circular padded tensor in the center
        top_bottom_padded[:, :, self.pad_width:height + self.pad_width, :] = circular_padded

        # Custom padding logic for top and bottom
        middle_index = width // 2
        for i in range(self.pad_width):
            # Top padding
            top_row = (self.pad_width - i-1) % height
            top_padding = torch.cat((circular_padded[:, :, top_row, middle_index:], circular_padded[:, :, top_row, :middle_index]), dim=-1)
            top_padding = top_padding.reshape(batch_size, channels, 1, -1)

            # Assign top padding
            top_bottom_padded[:, :, i, :] = top_padding[:, :, 0, :]

            # Bottom padding
            bottom_row = (height - i - 1) % height
            bottom_padding = torch.cat((circular_padded[:, :, bottom_row, middle_index:], circular_padded[:, :, bottom_row, :middle_index]), dim=-1)
            bottom_padding = bottom_padding.reshape(batch_size, channels, 1, -1)

            # Assign bottom padding
            top_bottom_padded[:, :, height + self.pad_width + i, :] = bottom_padding[:, :, 0, :]

        return top_bottom_padded

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        kernels = [3, 5, 7]
        self.pad = GeoCyclicPadding(pad_width=3)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=0, groups=dim) 
        self.se = SELayer(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pad(x)
        x = self.dwconv(x)
        x = self.se(x)  # Apply SE layer after dwconv
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

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=67, num_classes=67, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=1, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=1e-6) for j in range(depths[i])]
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
        x = self.forward_features(x)
        x = self.head(x)
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
        # Using the modified ConvNeXt as the core model
        self.core_model = ConvNeXt(in_chans=self.in_chans, num_classes=self.out_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate)

    def forward(self, x):
        x = self.core_model(x)
        return x


if __name__ == "__main__":
    # Example of creating a model and emphasizing specific channels
    model = KARINA()  # Update this line if you need to pass any specific parameters
    sample = torch.randn(1, 67, 72, 144)  # Example input tensor
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
