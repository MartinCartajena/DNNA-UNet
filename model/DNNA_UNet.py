import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size[0] // 2, bias=conv_bias)
        self.norm1 = norm_op(out_channels, **norm_op_kwargs)
        self.nonlin1 = nonlin(**nonlin_kwargs)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size[0] // 2, bias=conv_bias)
        self.norm2 = norm_op(out_channels, **norm_op_kwargs)
        self.nonlin2 = nonlin(**nonlin_kwargs)

        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=conv_bias) if in_channels != out_channels or stride != (1, 1, 1) else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlin2(x + residual)
        return x


class DNNA_UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_stages = 5
        self.features_per_stage = [32, 64, 128, 256, 320]
        self.kernel_sizes = [(3, 3, 3)] * self.n_stages
        self.strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)]
        self.n_blocks_per_stage = [1, 3, 4, 6, 6]
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {"eps": 1e-5, "affine": True}
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {"inplace": True}
        self.conv_bias = True

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1
        for stage in range(self.n_stages):
            stage_blocks = nn.Sequential(
                *[
                    ResidualBlock3D(
                        in_channels if block_idx == 0 else self.features_per_stage[stage],
                        self.features_per_stage[stage],
                        self.kernel_sizes[stage],
                        self.strides[stage] if block_idx == 0 else (1, 1, 1),
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                        self.conv_bias,
                    )
                    for block_idx in range(self.n_blocks_per_stage[stage])
                ]
            )
            self.encoder_blocks.append(stage_blocks)
            in_channels = self.features_per_stage[stage]

        # Bottleneck
        self.bottleneck = ResidualBlock3D(
            self.features_per_stage[-1],
            self.features_per_stage[-1],
            self.kernel_sizes[-1],
            (1, 1, 1),
            self.norm_op,
            self.norm_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.conv_bias,
        )

    def forward(self, x):
        # Encoder forward
        for stage_blocks in self.encoder_blocks:
            x = stage_blocks(x)
        # Bottleneck
        x = self.bottleneck(x)
        return x



