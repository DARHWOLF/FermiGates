from __future__ import annotations

import torch
import torch.nn as nn

from fermigates.base import BaseFermiClassifier
from fermigates.layers.conv_layers import FermiGatedConv2d
from fermigates.layers.linear_layers import FermiGatedLinear


class FermiConvClassifier(BaseFermiClassifier):
    """Compact CNN classifier with Fermi-gated convolution and linear head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (32, 64, 96),
        dropout: float = 0.1,
        init_mu: float = 0.0,
        init_T: float = 1.0,
    ) -> None:
        super().__init__(num_classes=num_classes)
        c1, c2, c3 = channels
        self.conv1 = FermiGatedConv2d(in_channels, c1, kernel_size=3, padding=1, init_mu=init_mu, init_T=init_T)
        self.conv2 = FermiGatedConv2d(c1, c2, kernel_size=3, padding=1, init_mu=init_mu, init_T=init_T)
        self.conv3 = FermiGatedConv2d(c2, c3, kernel_size=3, padding=1, init_mu=init_mu, init_T=init_T)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)
        self.norm3 = nn.BatchNorm2d(c3)

        self.dropout = nn.Dropout(dropout)
        self.classifier = FermiGatedLinear(c3, num_classes, init_mu=init_mu, init_T=init_T)

    def logits(self, x: torch.Tensor, return_masks: bool = False):
        masks: dict[str, torch.Tensor] = {}

        x, p = self.conv1(x)
        masks["conv1"] = p
        x = self.pool(torch.relu(self.norm1(x)))

        x, p = self.conv2(x)
        masks["conv2"] = p
        x = self.pool(torch.relu(self.norm2(x)))

        x, p = self.conv3(x)
        masks["conv3"] = p
        x = torch.relu(self.norm3(x))

        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        x = self.dropout(x)

        logits, p = self.classifier(x)
        masks["classifier"] = p

        if return_masks:
            return logits, masks
        return logits
