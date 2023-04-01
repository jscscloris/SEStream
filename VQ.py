from numpy import diff
import torch
from torch import nn
from torch.nn import functional as F
import distributed as dist_fn
from einops import rearrange, repeat
import random

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
        dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means.permute(1,0), bins


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5,kmeans_init = True,kmeans_iters = 10,threshold_ema_dead_code = 2):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.kmeans_iters=kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.n_embed, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
    
    def replace(self, samples, mask):
        a=sample_vectors(samples, self.n_embed).permute(1,0)
        modified_codebook = torch.where(
            mask[None,...],
            a,
            self.embed
        )
        self.embed.data.copy_(modified_codebook)


    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask = expired_codes)

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) 
        self.init_embed_(flatten)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1) 
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1]) 
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0) 
            embed_sum = flatten.transpose(0, 1) @ embed_onehot 

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(input)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResidualVQ(nn.Module):
    def __init__(self,n_q,dim, n_embed,decay=0.99, eps=1e-5):
        super().__init__()
        self.pre=Quantize(dim, n_embed, decay=decay, eps=eps)
        self.vq=nn.ModuleList()
        for i in range(n_q-1):
            self.vq.append(Quantize(dim, n_embed, decay=decay, eps=eps))

        self.n_q=n_q
        
    def forward(self, input,R=0):
        residual=input 
        quantized,diff,_=self.pre(input)
        residual=residual-quantized
        if R==0:
            train_vq=random.randint(3,self.n_q) - 1
        else:
            train_vq=int(R / 3 * 4)-1
        for vq in self.vq[:train_vq]:
            res,dif,_=vq(residual)
            quantized=quantized+res
            residual=residual-res
            diff=diff+dif      
        diff/=train_vq

        return quantized, diff