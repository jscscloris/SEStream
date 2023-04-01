from tkinter import N
import torch
from torch import nn
from torch.nn import functional as F
import modules
import VQ


SLICE=8640

class FiLM(nn.Module):

    def __init__(self, zdim, maskdim):
        super(FiLM, self).__init__()

        self.gamma = nn.Linear(zdim, maskdim)   # s
        self.beta = nn.Linear(zdim, maskdim)    # t
        stride=2*4*5*8
        self.down_sample=modules.CausalConv1d(2, 2, stride*2, stride)   

    def forward(self, x, z):
        z=self.down_sample(z).permute(0,2,1).contiguous() #(bs,frame,2)
        z=z.view(-1,2) #(bs*frame,2)
        gamma = self.gamma(z)
        beta = self.beta(z)      #(bs*frame,D)
        gamma=gamma.view(x.shape[0],x.shape[2],x.shape[1]) #(bs,frame,D)
        beta=beta.view(x.shape[0],x.shape[2],x.shape[1])

        x = gamma.permute(0,2,1) * x + beta.permute(0,2,1)

        return x

class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.blocks=nn.ModuleList()
        self.blocks.append(modules.CausalConv1d(in_channels=1, out_channels=C, kernel_size=7, stride=1))
        self.blocks.append(modules.EncoderBlock(in_channels=C,out_channels=2*C,stride=2))
        self.blocks.append(modules.EncoderBlock(in_channels=2*C,out_channels=4*C,stride=4))
        self.blocks.append(modules.EncoderBlock(in_channels=4*C,out_channels=8*C,stride=5))
        self.blocks.append(modules.EncoderBlock(in_channels=8*C,out_channels=16*C,stride=8))
        self.blocks.append(nn.ELU())
        self.blocks.append(modules.CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3, stride=1))    

    def forward(self,x):
        for l in self.blocks:
            x = l(x) #(batch_size,D,frame)
        x=torch.tanh(x)
        return x

class Encoder_denoise(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.blocks=nn.ModuleList()
        self.blocks.append(modules.CausalConv1d(in_channels=1, out_channels=C, kernel_size=7, stride=1))
        self.blocks.append(modules.EncoderBlock(in_channels=C,out_channels=2*C,stride=2))
        self.blocks.append(modules.EncoderBlock(in_channels=2*C,out_channels=4*C,stride=4))
        self.blocks.append(modules.EncoderBlock(in_channels=4*C,out_channels=8*C,stride=5))
        self.blocks.append(modules.EncoderBlock(in_channels=8*C,out_channels=16*C,stride=8))
        self.blocks.append(nn.ELU())
        self.blocks.append(modules.CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3, stride=1))  
        self.film_layer = FiLM(2, D)

    def forward(self,x,denoise):
        for l in self.blocks:
            x = l(x) #(batch_size,D,frame)
        x=self.film_layer(x,denoise)
        x=torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.blocks=nn.ModuleList()
        self.blocks.append(modules.CausalConv1d(in_channels=D, out_channels=16*C, kernel_size=7, stride=1))
        self.blocks.append(modules.DecoderBlock(in_channels=16*C,out_channels=8*C,stride=8))
        self.blocks.append(modules.DecoderBlock(in_channels=8*C,out_channels=4*C,stride=5))
        self.blocks.append(modules.DecoderBlock(in_channels=4*C,out_channels=2*C,stride=4))
        self.blocks.append(modules.DecoderBlock(in_channels=2*C,out_channels=C,stride=2))
        self.blocks.append(nn.ELU())
        self.blocks.append(modules.CausalConv1d(in_channels=C, out_channels=1, kernel_size=7, stride=1))    

    def forward(self,x):
        for l in self.blocks:
            x = l(x)
        x=torch.tanh(x)
        return x


class SoundStream(nn.Module):
    def __init__(self, C, D, n_q, codebook_size,if_vq):
        super().__init__()

        self.encoder = Encoder(C=C, D=D)
        self.quantize = VQ.ResidualVQ(n_q=n_q,dim=D, n_embed=codebook_size)
        self.decoder = Decoder(C=C, D=D)
        self.if_vq=if_vq

    def forward(self, x,R):
        e = self.encoder(x) #(batch,D,T)
        if self.if_vq:
          q,commit_loss=self.quantize(e.permute(0,2,1),R)#(batch,T,D)
          e=q.permute(0,2,1)#(batch,D,T)
        else:
          commit_loss=0.0
        o = self.decoder(e)
        return o,commit_loss

class SoundStream_denoise(nn.Module):
    def __init__(self, C, D, n_q, codebook_size,if_vq):
        super().__init__()

        self.encoder = Encoder_denoise(C=C, D=D)
        self.quantize = VQ.ResidualVQ(n_q=n_q,dim=D, n_embed=codebook_size)
        self.decoder = Decoder(C=C, D=D)
        self.if_vq=if_vq

    def forward(self, x,denoise,R=0):
        e = self.encoder(x,denoise) #(batch,D,T)
        if self.if_vq:
          q,commit_loss=self.quantize(e.permute(0,2,1),R)#(batch,T,D)
          e=q.permute(0,2,1)#(batch,D,T)
        else:
          commit_loss=0.0
        o = self.decoder(e)
        return o,commit_loss


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.pre = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=7)
    self.mids = nn.ModuleList()
    self.mids.append(nn.Conv1d(in_channels=16, out_channels=64, kernel_size=41, stride=4, groups=4, padding=20))
    self.mids.append(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=41, stride=4, groups=16, padding=20))
    self.mids.append(nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=41, stride=4, groups=64, padding=20))
    self.mids.append(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=4, groups=256, padding=20))
    self.mids.append(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2))
    self.post = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)

    nn.utils.weight_norm(self.pre)
    for l in self.mids:
      nn.utils.weight_norm(l)
    nn.utils.weight_norm(self.post)

  def forward(self, x):
    rets = []
    x = self.pre(x)
    x = F.leaky_relu(x)
    rets.append(x)
    for l in self.mids:
      x = l(x)
      x = F.leaky_relu(x)
      rets.append(x)
    x = self.post(x)
    rets.append(x)
    return x, rets


  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre)
    for l in self.mids:
      nn.utils.remove_weight_norm(l)
    nn.utils.remove_weight_norm(self.post)


class MultiScaleDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.discs = nn.ModuleList()
    for _ in range(3):
      self.discs.append(Discriminator())
    self.poolings = nn.ModuleList()
    self.poolings.append(nn.AvgPool1d(kernel_size=4, stride=2, padding=2))
    self.poolings.append(nn.AvgPool1d(kernel_size=4, stride=2, padding=2))

  def forward(self, x):
    ys = []
    rets = []
    down=[]
    down.append(x)
    for i, l in enumerate(self.discs):
      if i > 0:
        x = self.poolings[i-1](x)
        down.append(x)
      y, ret = l(x)
      ys.append(y)
      rets.extend(ret)
    return ys, rets,down


  def remove_weight_norm(self):
    for l in self.discs:
      l.remove_weight_norm()



class STFTDiscriminator(nn.Module):
  def __init__(self, C):
    super().__init__()
    W=1024
    F=W/2
    self.pre = nn.Conv2d(in_channels=2, out_channels=C, kernel_size=(7,7),padding='same')
    self.blocks=nn.ModuleList()
    self.blocks.append(modules.STFTUnit(in_channels=C,N=C,m=2,s_t=1,s_f=2))
    self.blocks.append(modules.STFTUnit(in_channels=2*C,N=2*C,m=2,s_t=2,s_f=2))
    self.blocks.append(modules.STFTUnit(in_channels=4*C,N=4*C,m=1,s_t=1,s_f=2))
    self.blocks.append(modules.STFTUnit(in_channels=4*C,N=4*C,m=2,s_t=2,s_f=2))
    self.blocks.append(modules.STFTUnit(in_channels=8*C,N=8*C,m=1,s_t=1,s_f=2))
    self.blocks.append(modules.STFTUnit(in_channels=8*C,N=8*C,m=2,s_t=2,s_f=2))
    self.post = nn.Conv2d(in_channels=16*C, out_channels=1, kernel_size=(1,int(F/2**6)))

    nn.utils.weight_norm(self.pre)
    nn.utils.weight_norm(self.post)

  def forward(self, x):
    rets=[]
    x = self.pre(x)
    rets.append(x)
    for l in self.blocks:
        x = l(x)
        rets.append(x)
    x = self.post(x)
    rets.append(x)
    return x,rets

  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre)
    nn.utils.remove_weight_norm(self.post)


