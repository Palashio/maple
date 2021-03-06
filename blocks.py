import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, filter_num=64, in_channels=1, dropout=False, dropout_rate=0.3):

        super(EncoderBlock, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.filter_num,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm2d(num_features=self.filter_num)
        self.bn_op_2 = nn.InstanceNorm2d(num_features=self.filter_num)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout2d(x, p=self.dropout_rate)

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout2d(x, p=self.dropout_rate)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False, dropout_rate=0.3):

        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        # Upsample by interpolation followed by a 3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='bilinear',
                                                               align_corners=True),

                                                   nn.Conv2d(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             kernel_size=3,
                                                             padding=1)
                                                   )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_tranposed = nn.ConvTranspose2d(in_channels=self.in_channels,
                                                      out_channels=self.in_channels,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1)

        self.down_sample = EncoderBlock(in_channels=self.in_channels+self.concat_layer_depth,
                                        filter_num=self.filter_num,
                                        dropout=self.dropout,
                                        dropout_rate=self.dropout_rate)

    def forward(self, x, skip_layer):
        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_tranposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out)
        return out


class EncoderBlock3D(nn.Module):
    def __init__(self, filter_num=64, in_channels=1, dropout=False):

        super(EncoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv3d(in_channels=self.filter_num,
                               out_channels=self.filter_num*2,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm3d(num_features=self.filter_num)
        self.bn_op_2 = nn.InstanceNorm3d(num_features=self.filter_num*2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)

        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        return x


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False):

        super(DecoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout

        # Upsample by interpolation followed by a 3x3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='nearest'),

                                                   nn.Conv3d(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             kernel_size=3,
                                                             padding=1)
                                                   )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1)

        if self.dropout is True:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                             nn.InstanceNorm3d(
                                                 num_features=self.filter_num),

                                             nn.LeakyReLU(),

                                             nn.Dropout3d(p=0.3),

                                             nn.Conv3d(in_channels=self.filter_num,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                             nn.InstanceNorm3d(
                                                 num_features=self.filter_num),

                                             nn.LeakyReLU(),

                                             nn.Dropout3d(p=0.3))
        else:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                             nn.InstanceNorm3d(
                                                 num_features=self.filter_num),

                                             nn.LeakyReLU(),

                                             nn.Conv3d(in_channels=self.filter_num,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                             nn.InstanceNorm3d(
                                                 num_features=self.filter_num),

                                             nn.LeakyReLU())

    def forward(self, x, skip_layer):

        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_transposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out)
        return out
