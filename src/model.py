import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=dilation,
        )
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=dilation,
        )
        self.norm_layer_1 = nn.BatchNorm2d(out_channels)
        self.norm_layer_2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        out = self.conv_1(x)
        out = self.norm_layer_1(out)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.norm_layer_2(out)
        out += skip_connection
        out = self.activation(out)
        return out


class ResidualBlockIncreaseDims(ResidualBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
        )
        self.skip_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            bias=False,
        )
        self.skip_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        out = self.conv_1(x)
        out = self.norm_layer_1(out)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.norm_layer_2(out)

        # increasing dims of the skip connection
        skip_connection = self.skip_conv(skip_connection)
        skip_connection = self.skip_bn(skip_connection)

        out += skip_connection
        out = self.activation(out)
        return out


class ResNetCustom(nn.Module):
    INPUT_SHAPE: tuple[int, int] = (28, 28)

    def __init__(
        self,
        n_classes: int,
        input_channels: int = 1,
        blocks_size: tuple[int, ...] = (64, 64, 128),
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, blocks_size[0], 7, stride=2)
        self.bn1 = nn.BatchNorm2d(blocks_size[0])
        self.activation = nn.ReLU()

        self.conv2_1 = ResidualBlock(blocks_size[0], blocks_size[1])
        self.conv2_2 = ResidualBlock(blocks_size[1], blocks_size[1])

        self.conv3_1 = ResidualBlockIncreaseDims(
            blocks_size[0],
            blocks_size[2],
        )
        self.conv3_2 = ResidualBlock(blocks_size[2], blocks_size[2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(blocks_size[-1], n_classes)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv1 28x28 -> 28x28
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # conv2_x 28x28 -> 14x14
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        # conv3_x 14x14 -> 7x7
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # final layers
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.final_activation(x)
        return x
