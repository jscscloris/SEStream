from tkinter import N
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels,out_channels,dilation):
        super().__init__()
        self.res=nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=1,dilation=dilation),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        )
    
    def forward(self,x):
        return x+self.res(x)

class STFTUnit(nn.Module):
    def __init__(self, in_channels,N,m,s_t,s_f):
        super().__init__()
        self.res=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=N,kernel_size=(3,3),padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=N,out_channels=m*N,kernel_size=(s_t+2,s_f+2),stride=(s_t, s_f),padding=(1,1))
        )
        self.skip_connection = nn.Conv2d(in_channels=in_channels,out_channels=m*N,kernel_size=(s_t, s_f), stride=(s_t, s_f))
        
        nn.utils.weight_norm(self.res[1])
        nn.utils.weight_norm(self.res[3])
        nn.utils.weight_norm(self.skip_connection)

    def forward(self,x):
        return self.res(x)+self.skip_connection(x)
    
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.res[1])
        nn.utils.remove_weight_norm(self.res[3])
        nn.utils.remove_weight_norm(self.skip_connection)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        kernel_size=2*stride
        self.layer= nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels//2,dilation=1),
            ResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2,dilation=3),
            ResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2,dilation=9),
            nn.ELU(),
            CausalConv1d(out_channels//2, out_channels, kernel_size, stride)     
        )
    def forward(self, x):
        return self.layer(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        kernel_size=2*stride
        self.layer= nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(in_channels, out_channels, kernel_size, stride),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,dilation=1),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,dilation=3),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,dilation=9)
        )

    def forward(self,x):
        return self.layer(x)


        
