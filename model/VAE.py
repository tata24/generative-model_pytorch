import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # (96-5+2*1)/3+1=32  3*96*96->64*32*32
            nn.Conv2d(in_channels, 64, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (32-4+2*1)/2+1=16 64*32*32->128*16*16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (16-4+2*1)/2+1=8  128*16*16->256*8*8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (8-4+2*1)/2+1=4 256*8*8->512*4*4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # (4-4+2*1)/2+0 = 1 512*4*4->100*1*1
        self.mu_vec = nn.Conv2d(512, out_channels, 4, 1, 0, bias=False)
        # (4-4+2*1)/2+0 = 1 512*4*4->100*1*1
        self.log_var_vec = nn.Conv2d(512, out_channels, 4, 1, 0, bias=False)
        self.decoder = nn.Sequential(
            # (1-1)*1+0-2*0+4=4   100*1*1->512*4*4
            nn.ConvTranspose2d(out_channels, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # (4-1)*2+0-2*1+4=8   512*4*4->256*8*8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (8-1)*2+0-2*1+4=16  256*8*8->128*16*16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (16-1)*2+0-2*1+4=32  128*16*16->64*32*32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (32-1)*3+0-2*1+5=96  64*32*32->3*96*96
            nn.ConvTranspose2d(64, in_channels, 5, 3, 1, bias=False),
            # 将范围限制在（-1，1）
            nn.Tanh()
        )

    # Z = mu + eps * sigma
    def reparameterization(self, mu, log_var):
        sigma = log_var.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(sigma.size()).normal_()
        else:
            eps = torch.FloatTensor(sigma.size()).normal_()
        return eps.mul_(sigma).add_(mu)

    def forward(self, input):
        encoder_out = self.encoder(input)
        mu = self.mu_vec(encoder_out)
        log_var = self.log_var_vec(encoder_out)
        Z = self.reparameterization(mu, log_var)
        return self.decoder(Z), mu, log_var














