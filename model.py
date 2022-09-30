# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch import nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "Discriminator", "Generator", "ContentLoss",
    "discriminator", "dsgan", "content_loss",

]


# Cropy from `https://github.com/ManuelFritsche/real-world-sr/blob/master/dsgan/model.py`
class _GaussianFilter(nn.Module):
    def __init__(self, kernel_size: int = 5, stride: int = 1, padding: int = 4) -> None:
        super(_GaussianFilter, self).__init__()
        # Initialize gaussian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depth-wise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, (kernel_size, kernel_size), (stride, stride), (padding, padding),
                                         groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.gaussian_filter(x)

        return out


class FilterLowFrequencies(nn.Module):
    def __init__(
            self,
            recursions: int = 1,
            kernel_size: int = 5,
            stride: int = 1,
            use_padding: bool = True,
            avg_count_include_pad: bool = True,
            use_gaussian: bool = False
    ) -> None:
        super(FilterLowFrequencies, self).__init__()
        self.recursions = recursions

        if use_padding:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0

        if use_gaussian:
            self.filter = _GaussianFilter(kernel_size, stride, padding)
        else:
            self.filter = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=avg_count_include_pad)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.recursions):
            x = self.filter(x)

        return x


class FilterHighFrequencies(nn.Module):
    def __init__(
            self,
            recursions: int = 1,
            kernel_size: int = 5,
            stride: int = 1,
            use_padding: bool = True,
            avg_count_include_pad: bool = True,
            use_gaussian: bool = False,
    ) -> None:
        super(FilterHighFrequencies, self).__init__()
        self.filter = FilterLowFrequencies(1, kernel_size, stride, use_padding, avg_count_include_pad, use_gaussian)
        self.recursions = recursions

    def forward(self, x: Tensor) -> Tensor:
        if self.recursions > 1:
            for _ in range(self.recursions - 1):
                x = self.filter(x)
        out = x - self.filter(x)

        return out


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.rb(x)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            recursions: int = 1,
            kernel_size: int = 5,
            stride: int = 1,
            filter_high_frequencies: bool = True,
            use_gaussian: bool = False,
    ) -> None:
        super(Discriminator, self).__init__()
        if filter_high_frequencies:
            self.filter = FilterHighFrequencies(recursions, kernel_size, stride, True, False, use_gaussian)
        else:
            self.filter = nn.Identity()

        self.disc_model = nn.Sequential(
            nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 1, (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        if self.filter is not None:
            x = self.filter(x)

        out = self.disc_model(x)

        if y is not None:
            diff = self.filter(y)
            diff = self.disc_model(diff)
            diff = torch.mean(diff, 0, keepdim=True)
            out -= diff

        # Keep -1 ~ 1
        out = torch.sigmoid(out)

        return out


class Generator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_blocks: int = 8,
    ) -> None:
        super(Generator, self).__init__()

        # The first layer of convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        # Feature extraction backbone network
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        # Keep -1 ~ 1
        out = torch.sigmoid(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F.mse_loss(sr_feature, gt_feature)

        return loss


def discriminator(**kwargs: Any) -> Discriminator:
    model = Discriminator(**kwargs)

    return model


def dsgan(**kwargs: Any) -> Generator:
    model = Generator(**kwargs)

    return model


def content_loss(feature_model_extractor_nodes,
                 feature_model_normalize_mean,
                 feature_model_normalize_std) -> ContentLoss:
    content_loss = ContentLoss(feature_model_extractor_nodes,
                               feature_model_normalize_mean,
                               feature_model_normalize_std)

    return content_loss
