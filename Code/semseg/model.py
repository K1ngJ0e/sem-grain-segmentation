"""U-Net++ / ResNet18 with SE applied to encoder feature maps."""

import torch.nn as nn
import segmentation_models_pytorch as smp


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.fc(self.pool(inputs))


class UnetPlusPlusEncoderSE(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Fail explicitly if requested ImageNet initialization cannot be loaded.
        self.net = smp.UnetPlusPlus(
            encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=1, activation=None,
        )
        self.se = nn.ModuleList([
            nn.Identity() if index == 0 else SEBlock(channels)
            for index, channels in enumerate(self.net.encoder.out_channels)
        ])

    def forward(self, inputs):
        features = [module(value) for module, value in zip(self.se, self.net.encoder(inputs))]
        # segmentation-models-pytorch 0.5.0 decoder API.
        return self.net.segmentation_head(self.net.decoder(features))
