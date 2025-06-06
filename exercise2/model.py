# Adapted from : https://github.com/dome272/Diffusion-Models-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F



def pos_encoding(t, channels, device):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc



class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, img_size=16, c_in=3, c_out=3, time_dim=256, device="cpu", channels=32, num_classes=None):
        '''If num_classes is None then it is a standard UNet
           Expects one-hot encoded classes '''
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2,  emb_dim=time_dim)
        self.sa1 = SelfAttention(channels*2, img_size//2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        
        self.sa2 = SelfAttention(channels*4, img_size // 4)
        self.down3 = Down(channels*4, channels*4,  emb_dim=time_dim)
        self.sa3 = SelfAttention(channels*4, img_size // 8)

        self.bot1 = DoubleConv(channels*4, channels*8)
        self.bot2 = DoubleConv(channels*8, channels*8)
        self.bot3 = DoubleConv(channels*8, channels*4)

        self.up1 = Up(channels*8, channels*2,  emb_dim=time_dim)
        self.sa4 = SelfAttention(channels*2, img_size // 4)
        self.up2 = Up(channels*4, channels,  emb_dim=time_dim)
        self.sa5 = SelfAttention(channels, img_size // 2)
        self.up3 = Up(channels*2, channels,  emb_dim=time_dim)
        self.sa6 = SelfAttention(channels, img_size)
        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

        if num_classes is not None:
            # Project one-hot encoded labels to the time embedding dimension 
            # Implement it as a 2-layer MLP with a GELU activation in-between
            self.label_emb = nn.Sequential(nn.Linear(num_classes, time_dim),
                                           nn.GELU(),
                                           nn.Linear(time_dim, time_dim))
            

    def forward(self, x, t, y=None):

        t = t.unsqueeze(-1).type(torch.float)
        t = pos_encoding(t, self.time_dim, self.device)

        if y is not None:
            # Add label and time embeddings together
            t += self.label_emb(y)
            
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output

class Classifier(nn.Module):
    def __init__(self, img_size=16, c_in=3, labels=5, time_dim=256, device="cuda", channels=32):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2,  emb_dim=time_dim)
        self.sa1 = SelfAttention(channels*2, img_size//2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        
        self.sa2 = SelfAttention(channels*4, img_size // 4)
        self.down3 = Down(channels*4, channels*4,  emb_dim=time_dim)
        self.sa3 = SelfAttention(channels*4, img_size // 8)

        self.bot1 = DoubleConv(channels*4, channels*8)
        self.bot2 = DoubleConv(channels*8, channels*8)
        self.bot3 = DoubleConv(channels*8, channels*4)

        self.outc = nn.Linear(channels*4*2*2, labels)


    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = pos_encoding(t, self.time_dim, self.device)
            
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        output = self.outc(x4.flatten(1))
        return output