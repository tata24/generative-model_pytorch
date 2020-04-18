import torch.nn as nn


# 生成器
# 转置卷积输出尺寸计算公式
# output=(input-1)*stride+output_padding-2*padding+kernel_size
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # (1-1)*1+0-2*0+4=4   100*1*1->512*4*4
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # (4-1)*2+0-2*1+4=8   512*4*4->256*8*8
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # (8-1)*2+0-2*1+4=16  256*8*8->128*16*16
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # (16-1)*2+0-2*1+4=32  128*16*16->64*32*32
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # (32-1)*3+0-2*1+5=96  64*32*32->3*96*96
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=5,
                stride=3,
                padding=1,
                bias=False),
            # 将范围限制在（-1，1）
            nn.Tanh()
        )

    def forward(self, input):
        return self.generator(input)


# 判别器
# 卷积输出尺寸计算公式
# output=(input−kernel_size+2*padding)/stride+1
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # (96-5+2*1)/3+1=32  3*96*96->64*32*32
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                stride=3,
                padding=1,
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (32-4+2*1)/2+1=16 64*32*32->128*16*16
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (16-4+2*1)/2+1=8  128*16*16->256*8*8
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (8-4+2*1)/2+1=4 256*8*8->512*4*4
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.discriminator(input)