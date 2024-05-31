import sys
sys.path.append('/home/ubuntu/jesstoh/image-segmentation-in-dark/diffusion_model/model')

import torch
import torch.nn as nn
from blocks import *

class Unet(nn.Module):
    def __init__(
        self,
        in_channel=6, # Concat input image and condition image channels, ie. 2 x 3
        out_channel=3, # Output channel for image
        inner_channel=32,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
    ):
        super().__init__()
        # self.init_conv = nn.Conv2d(in_channel, inner_channel, 7, padding=3)
        # dims = [*map(lambda m: inner_channel * m, dim_mults)]
        dims = [in_channel, *map(lambda m: inner_channel * m, dim_mults)]
        # Up and downsampling channels
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        if with_time_emb:
            time_dim = inner_channel * 4
            self.time_mlp = nn.Sequential(
                TimeEmbeddings(inner_channel),
                nn.Linear(inner_channel, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None


        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim),
                        ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ConvNextBlock(inner_channel, inner_channel), 
            nn.Conv2d(inner_channel, out_channel, 1)
        )

    def forward(self, x, time):
        # x = self.init_conv(x)

        t = None
        if exists(self.time_mlp) and exists(time):
            t = self.time_mlp(time)

        # Keep layer in downsample to connect to upsample.
        h = []

        # Downsample
        for convnext1, convnext2, attn, downsample in self.downs:
            x = convnext1(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for convnext1, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # Residual connection from downsampling layer
            x = convnext1(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)