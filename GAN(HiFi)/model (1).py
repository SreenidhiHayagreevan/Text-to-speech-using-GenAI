import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HIFIGANConfig:

    upsample_rates: Tuple = (8,8,2,2)
    upsample_kernel_sizes: Tuple = (16,16,4,4)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: Tuple = (3,7,11)
    resblock_dilation_sizes: Tuple[Tuple[int]] = ((1,3,5), (1,3,5), (1,3,5))
    
    mpd_periods: Tuple = (2,3,5,7,11)
    msd_num_downsamples: int = 2

    num_mels: int = 80

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        with torch.no_grad():
            m.weight.normal_(mean, std)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1,3,5)):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.d1, self.d2, self.d3 = dilation

        self.conv_stack_1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size,
                        stride=1, 
                        dilation=self.d1, 
                        padding="same"
                    )
                ),

                weight_norm(
                    nn.Conv1d(
                        in_channels=channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size,
                        stride=1, 
                        dilation=self.d2, 
                        padding="same"
                    )
                ),

                weight_norm(
                    nn.Conv1d(
                        in_channels=channels, 
                        out_channels=channels, 
                        kernel_size=kernel_size,
                        stride=1, 
                        dilation=self.d3, 
                        padding="same"
                    )
                )

            ]
        )

        self.conv_stack_1.apply(init_weights)

        self.conv_stack_2 = nn.ModuleList(
                [
                    weight_norm(
                        nn.Conv1d(
                            in_channels=channels, 
                            out_channels=channels, 
                            kernel_size=kernel_size,
                            stride=1, 
                            padding="same"
                        )
                    ),

                    weight_norm(
                        nn.Conv1d(
                            in_channels=channels, 
                            out_channels=channels, 
                            kernel_size=kernel_size,
                            stride=1, 
                            padding="same"
                        )
                    ),

                    weight_norm(
                        nn.Conv1d(
                            in_channels=channels, 
                            out_channels=channels, 
                            kernel_size=kernel_size,
                            stride=1, 
                            padding="same"
                        )
                    )

                ]
            )
        
        self.conv_stack_2.apply(init_weights)

    def forward(self, x):

        for proj1, proj2  in zip(self.conv_stack_1, self.conv_stack_2):

            residual = x

            x = F.leaky_relu(x, negative_slope=0.1)
            x = proj1(x)

            x = F.leaky_relu(x, negative_slope=0.1)
            x = proj2(x)

            x = x + residual

        return x
    
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.config = config
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        print(f"Effective Upsample Rate: {torch.prod(torch.tensor(config.upsample_rates))}")

        ### Input Projection on Mel Spectrograms ###
        self.input_proj = weight_norm(
            nn.Conv1d(
                in_channels=config.num_mels, 
                out_channels=config.upsample_initial_channel, 
                kernel_size=7, 
                stride=1, 
                padding="same"
            )
        )

        ### Stack of Generator Blocks ###
        self.generator_blocks = nn.ModuleList()
        in_channels = config.upsample_initial_channel
        for upsample_rate, kernel_size in zip(config.upsample_rates, config.upsample_kernel_sizes):
            
            up_block = nn.ModuleList()
            out_channels = in_channels // 2
            
            ### Upsample Block ###
            up_block.append(
                    weight_norm(
                        nn.ConvTranspose1d(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=upsample_rate, 
                            padding=(kernel_size-upsample_rate)//2 
                        )
                    )
                )
            
            ### Stack of Residual Blocks that Follow Each Upsample (For multilevel generation) ### 
            for residual_kernel, residual_dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                up_block.append(
                    ResidualBlock(
                        channels=out_channels, 
                        kernel_size=residual_kernel, 
                        dilation=residual_dilation
                    )
                )

            self.generator_blocks.append(up_block)

            in_channels = out_channels

        self.output_proj = weight_norm(
            nn.Conv1d(
                in_channels=out_channels, 
                out_channels=1, 
                kernel_size=7, 
                stride=1, 
                padding="same"
            )
        )

        self.generator_blocks.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(self, x):

        ### Input Projection ###
        x = self.input_proj(x)

        ### Loop Through Generator Blocks ###
        for i in range(self.num_upsamples):
            
            ### Grab the Modules in the Blocks ###
            block = self.generator_blocks[i]
            upsample, residuals = block[0], block[1:]

            ### Upsample Spectrogram ###
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)

            ### Average Upsampled Result between all residual blocks ###
            x = torch.stack([residual(x) for residual in residuals]).mean(dim=0)

        ### Output Projection to WaveForm ###
        x = F.leaky_relu(x)
        x = self.output_proj(x)
        x = torch.tanh(x)

        return x

class PeriodicDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(PeriodicDiscriminator, self).__init__()
        
        self.period = period
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_block = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels=1, 
                        out_channels=32, 
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1), 
                        padding=(2,0)
                    )
                ), 

                weight_norm(
                    nn.Conv2d(
                        in_channels=32, 
                        out_channels=128, 
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1), 
                        padding=(2,0)
                    )
                ), 

                weight_norm(
                    nn.Conv2d(
                        in_channels=128, 
                        out_channels=512, 
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1), 
                        padding=(2,0)
                    )
                ), 

                weight_norm(
                    nn.Conv2d(
                        in_channels=512, 
                        out_channels=1024, 
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1), 
                        padding=(2,0)
                    )
                ), 

                weight_norm(
                    nn.Conv2d(
                        in_channels=1024, 
                        out_channels=1024, 
                        kernel_size=(kernel_size, 1),
                        stride=1, 
                        padding=(2,0)
                    )
                )
            ]
        )

        self.output_proj = weight_norm(
            nn.Conv2d(
                in_channels=1024, 
                out_channels=1, 
                kernel_size=(3,1),
                stride=1, 
                padding=(1,0)
            )
        )

    def forward(self, x):

        feature_maps = []
   
        batch_size, channels, seq_len = x.shape

        if seq_len % self.period != 0:
            n_pad = self.period - (seq_len % self.period)
            x = F.pad(x, (0,n_pad), "reflect")
            seq_len += n_pad

        x = x.reshape(batch_size, channels, seq_len//self.period, self.period)

        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.output_proj(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()

        self.config = config

        self.discriminators = nn.ModuleList(
            [PeriodicDiscriminator(p) for p in config.mpd_periods]
        )

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        for discrim in self.discriminators:
            real_out, real_feat_map = discrim(real)
            gen_out, gen_feat_map = discrim(gen)

            real_outs.append(real_out)
            gen_outs.append(gen_out)
            real_feat_maps.append(real_feat_map)
            gen_feat_maps.append(gen_feat_map)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps

class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ScaleDiscriminator, self).__init__()

        norm = weight_norm if not use_spectral_norm else spectral_norm

        self.conv_block = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        in_channels=1, 
                        out_channels=128, 
                        kernel_size=15, 
                        stride=1, 
                        padding=7
                    )
                ), 

                norm(
                    nn.Conv1d(
                        in_channels=128, 
                        out_channels=128, 
                        kernel_size=41, 
                        stride=2, 
                        groups=4, 
                        padding=20
                    )
                ), 

                norm(
                    nn.Conv1d(
                        in_channels=128, 
                        out_channels=256, 
                        kernel_size=41, 
                        stride=2, 
                        groups=16, 
                        padding=20
                    )
                ), 

                norm(
                    nn.Conv1d(
                        in_channels=256, 
                        out_channels=512, 
                        kernel_size=41, 
                        stride=4, 
                        groups=16, 
                        padding=20
                    )
                ), 
                norm(
                    nn.Conv1d(
                        in_channels=512, 
                        out_channels=1024, 
                        kernel_size=41, 
                        stride=4, 
                        groups=16, 
                        padding=20
                    )
                ), 
                norm(
                    nn.Conv1d(
                        in_channels=1024, 
                        out_channels=1024, 
                        kernel_size=41, 
                        stride=1, 
                        groups=16, 
                        padding=20
                    )
                ), 
                norm(
                    nn.Conv1d(
                        in_channels=1024, 
                        out_channels=1024, 
                        kernel_size=5, 
                        stride=1, 
                        padding=2
                    )
                )
            ]
        )

        self.output_proj = norm(
            nn.Conv1d(
                in_channels=1024, 
                out_channels=1, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        )
    
    def forward(self, x):

        feature_maps = []
        
        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.output_proj(x)
        feature_maps.append(x)

        return x, feature_maps

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()

        self.config = config

        self.discriminator = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=True)] + [ScaleDiscriminator() for _ in range(config.msd_num_downsamples)]
        )

        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(config.msd_num_downsamples)
            ]
        )

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        for i, discrim in enumerate(self.discriminator):

            if i != 0:
                real = self.meanpools[i-1](real)
                gen = self.meanpools[i-1](gen)

            real_out, real_feat_map = discrim(real)
            gen_out, gen_feat_map = discrim(gen)

            real_outs.append(real_out)
            real_feat_maps.append(real_feat_map)
            gen_outs.append(gen_out)
            gen_feat_maps.append(gen_feat_map)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps

class HIFIGAN(nn.Module):
    def __init__(self, config):
        super(HIFIGAN, self).__init__()
        self.generator = Generator(config)
        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator(config)

    def _get_generator_params(self):
        return self.generator.parameters()

    def _get_discriminator_params(self):
        return list(self.mpd.parameters()) + list(self.msd.parameters())
