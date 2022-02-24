import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, *, kernel_size, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        # stride无非就是1/2
        assert stride in [1, 2]
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.padding = padding
        self.in_channels = in_channels
        out = out_channels - in_channels

        # 论文中的右侧通路
        branch_main = [
            # 先 1*1卷积降低通道数量
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 下采样模块，stride=2就缩小1/2*1/2， stride=1不变， 深度可分离卷积
            # (input-kernel+2*padding)//stride + 1
            # if stride=2 => input / 2 向上取整
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # 恢复原有通道数量
            nn.Conv2d(mid_channels, out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)
        # 论文中d模块的左半部分， stride=2就下采样后通道数量翻倍，否则直接连接
        if stride == 2:
            branch_proj = [
                # 下采样 (input - kernel + 2 * padding) // stride + 1
                # (input - 9 + 8 ) // 2 + 1 = input / 2 向上取整
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                # 1*1卷积
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_spilt_1, x_spilt_2 = self.split_channels(old_x)
            return torch.cat((x_spilt_1, self.branch_main(x_spilt_2)), 1)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(old_x), self.branch_main(old_x)), 1)


    def split_channels(self, x):
        batch, channels, height, width = x.size()
        assert channels % 4 == 0
        x = torch.reshape(x, (batch, 2, channels//2, height, width))
        x = x.transpose(1, 2)
        x = torch.reshape(x, (batch, channels, height, width))
        x = torch.reshape(x, (batch, 2, channels // 2, height, width))
        x = x.transpose(0, 1)
        return x[0], x[1]
