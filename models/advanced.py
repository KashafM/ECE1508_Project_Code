"""Advanced attention-based U-Net variant."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseUNet
from .registry import register_model


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = self.conv_block(x)
        out = out + identity
        return self.relu(out)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = ResidualDoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        self.cbam = CBAM(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(x)
        attention_out = self.cbam(conv_out)
        pooled = self.pool(attention_out)
        return attention_out, pooled


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bilinear: bool = True,
        dropout_rate: float = 0.1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv_after_up = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            decoder_input_channels = in_channels // 2 + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_after_up = None
            decoder_input_channels = in_channels // 2 + skip_channels

        self.use_attention = use_attention
        if use_attention:
            F_int = max(skip_channels // 2, 1)
            self.attention_gate = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=F_int)

        self.conv = ResidualDoubleConv(decoder_input_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.conv_after_up is not None:
            x = self.conv_after_up(x)
        if self.use_attention:
            skip = self.attention_gate(x, skip)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AdvancedBrainTumorUNet(BaseUNet):
    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 5,
        base_features: int = 64,
        depth: int = 4,
        bilinear: bool = False,
        dropout_rate: float = 0.1,
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        self.depth = depth
        self.use_deep_supervision = use_deep_supervision

        self.input_conv = ResidualDoubleConv(n_channels, base_features, dropout_rate=dropout_rate)

        self.encoders = nn.ModuleList()
        in_channels = base_features
        for i in range(depth):
            out_channels = base_features * (2 ** (i + 1))
            self.encoders.append(EncoderBlock(in_channels, out_channels, dropout_rate=dropout_rate))
            in_channels = out_channels

        bottleneck_channels = base_features * (2 ** depth)
        factor = 2 if bilinear else 1
        self.bottleneck = ResidualDoubleConv(in_channels, bottleneck_channels // factor, dropout_rate=dropout_rate)

        self.decoders = nn.ModuleList()
        decoder_in_channels = bottleneck_channels // factor
        for i in range(depth - 1, -1, -1):
            skip_channels = base_features * (2 ** i) if i > 0 else base_features
            out_channels = skip_channels
            self.decoders.append(
                DecoderBlock(
                    decoder_in_channels,
                    skip_channels,
                    out_channels,
                    bilinear=bilinear,
                    dropout_rate=dropout_rate,
                    use_attention=True,
                )
            )
            decoder_in_channels = out_channels

        self.final_conv = nn.Conv2d(base_features, n_classes, kernel_size=1)

        if use_deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(depth - 1):
                in_ch = base_features * (2 ** (depth - 1 - i))
                self.deep_supervision_outputs.append(nn.Conv2d(in_ch, n_classes, kernel_size=1))
        else:
            self.deep_supervision_outputs = None

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        encoder_features = [x]
        for encoder in self.encoders:
            features, x = encoder(x)
            encoder_features.append(features)

        x = self.bottleneck(x)
        deep_outputs = []
        for i, decoder in enumerate(self.decoders):
            skip_connection = encoder_features[-(i + 2)]
            x = decoder(x, skip_connection)
            if self.use_deep_supervision and self.training and self.deep_supervision_outputs is not None and i < len(self.decoders) - 1:
                ds_out = self.deep_supervision_outputs[i](x)
                scale_factor = 2 ** (len(self.decoders) - i - 1)
                if scale_factor > 1:
                    ds_out = F.interpolate(ds_out, scale_factor=scale_factor, mode="bilinear", align_corners=True)
                deep_outputs.append(ds_out)

        output = self.final_conv(x)
        if self.use_deep_supervision and self.training and self.deep_supervision_outputs is not None:
            deep_outputs.append(output)
            return deep_outputs
        return output


register_model(
    name="advanced_unet",
    constructor=AdvancedBrainTumorUNet,
    metadata={
        "display_name": "Advanced Attention U-Net",
        "default_args": {
            "n_channels": 4,
            "n_classes": 5,
            "base_features": 64,
            "depth": 4,
            "bilinear": False,
            "dropout_rate": 0.1,
            "use_deep_supervision": True,
        },
        "description": "Residual, attention-augmented U-Net with optional deep supervision.",
    },
)
