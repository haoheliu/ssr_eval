import git
import sys

sys.path.append("/vol/research/dcase2022/sr_eval_vctk")
import os
import numpy as np

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        super(ConvBlock, self).__init__()

        self.activation = activation
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = act(self.bn1(self.conv1(x)), self.activation)
        x = act(self.bn2(self.conv2(x)), self.activation)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, downsample, activation, momentum
    ):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block(x)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, upsample, activation, momentum
    ):
        super(DecoderBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv_block2 = ConvBlock(
            out_channels * 2, out_channels, kernel_size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)

    def prune(self, x):
        """Prune the shape of x after transpose convolution."""
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        x = x[
            :,
            :,
            padding[0] : padding[0] - self.stride[0],
            padding[1] : padding[1] - self.stride[1],
        ]
        return x

    def forward(self, input_tensor, concat_tensor):
        x = act(self.bn1(self.conv1(input_tensor)), self.activation)
        # from IPython import embed; embed(using=False); os._exit(0)
        # x = self.prune(x)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        return x


class EncoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes1B, self).__init__()
        size = (3, 3)

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes1B, self).__init__()
        size = (3, 3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=stride,
            padding=(0, 0),
            output_padding=(0, 0),
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution."""
        if both:
            x = x[:, :, 0:-1, 0:-1]
        else:
            x = x[:, :, 0:-1, :]
        return x

    def forward(self, input_tensor, concat_tensor, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class EncoderBlockRes4B(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes4B, self).__init__()
        size = (3, 3)

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes4B(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes4B, self).__init__()
        size = (3, 3)
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=stride,
            padding=(0, 0),
            output_padding=(0, 0),
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes(
            out_channels, out_channels, size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution."""
        if both:
            x = x[:, :, 0:-1, 0:-1]
        else:
            x = x[:, :, 0:-1, :]
        return x

    def forward(self, input_tensor, concat_tensor, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn):
    """Initialize a GRU layer."""

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)


from torch.cuda import init


def act(x, activation):
    if activation == "relu":
        return F.relu_(x)

    elif activation == "leaky_relu":
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == "swish":
        return x * torch.sigmoid(x)

    else:
        raise Exception("Incorrect activation!")


class UNetResComplex_100Mb(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(UNetResComplex_100Mb, self).__init__()
        activation = "relu"
        momentum = 0.01

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blcoks}

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=channels * nsrc,
            out_channels=32,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlockRes(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            stride=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlockRes(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, sp):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # Batch normalization
        x = sp

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool
        )  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool
        )  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        output_dict = {"mel": x}
        return output_dict


if __name__ == "__main__":
    model = UNetResComplex_100Mb(channels=1)
    print(model(torch.randn((1, 1, 101, 128)))["mel"].size())
