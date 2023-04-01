import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.autograd as autograd
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import two_stage_LibriTTS
from stft_loss import MultiResolutionSTFTLoss
from spec_loss import MultiResolutionLoss
import models
import VQ

parser=argparse.ArgumentParser(description="the code of SEStream")
parser.add_argument("--batch_size",type=int,default=128,help="Batch size")
parser.add_argument("--num_gpu",type=int,default=1,help="number of gpu")
parser.add_argument("--epochs",type=int,default=1000,help="training epoch")
parser.add_argument("--start_epoch",type=int,default=0,help="start epoch")
parser.add_argument("--load_checkpoint",type=str,default=None,help="if load chechpoint")
parser.add_argument("--lambda_adv",type=int,default=1,help="lambda for adv loss")
parser.add_argument("--lambda_feat",type=int,default=100,help="lambda for feature loss")
parser.add_argument("--lambda_rec",type=int,default=30,help="lambda for rec loss")
parser.add_argument("--channel_g",type=int,default=32,help="number of channel for soundstream")
parser.add_argument("--channel_d",type=int,default=32,help="number of channnel for stft_disc")
parser.add_argument("--output_path",type=str,default="./libritts",help="the path for result")
parser.add_argument("--stage",type=int,default=1,help="the training stage of the model")
parser.add_argument("--target_bit",type=int,default=6,help="target bit rate. If 0, train with quantizer dropout")
parser.add_argument("--n_q",type=int,default=24,help="the number of VQs")
parser.add_argument("--n_embed",type=int,default=1024,help="the size of codebook")
parser.add_argument("--rec_type",type=int,default=1,help="the type of reconstruction loss")

args=parser.parse_args()

writer=SummaryWriter(os.path.join(args.output_path,'scalar'))

if args.num_gpu==0:
    device=torch.device("cpu")
else:
    device=torch.device("cuda:0")

encoder=models.Encoder(C=args.channel_g,D=256)
vq=VQ.ResidualVQ(n_q=args.n_q,dim=256, n_embed=args.n_embed)
decoder=models.Decoder(C=args.channel_g, D=256)

encoder.to(device)
vq.to(device)
decoder.to(device)

if args.stage==1:
    optimizer_g = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

elif args.stage==2:
    WINDOW_LENGTH=1024
    HOP_LENGTH=256
    wave_disc = models.MultiScaleDiscriminator().to(device)
    stft_disc = models.STFTDiscriminator(C=args.channel_d).to(device)
    optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4)
    encoder.load_state_dict(torch.load(os.path.join(args.output_path,"MMSE_encoder.pth"),map_location=device))
    vq.load_state_dict(torch.load(os.path.join(args.output_path,"MMSE_vq.pth"),map_location=device))

    if args.load_checkpoint!=None:
        decoder.load_state_dict(torch.load(os.path.join(args.output_path,args.load_checkpoint+"_decoder.pth"),map_location=device))
        stft_disc.load_state_dict(torch.load(os.path.join(args.output_path,"final_stft.pth"),map_location=device))
        wave_disc.load_state_dict(torch.load(os.path.join(args.output_path,"final_wave.pth"),map_location=device))
        print("load success!")
    
    optimizer_g = optim.Adam(decoder.parameters(), lr=1e-4)


def collate_fn(batch):
    lengths = torch.tensor([elem[0].shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch[0], batch_first=True), lengths

train_dataset=two_stage_LibriTTS(audio_dir="train_full_file.json")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2,shuffle= True)
sr = train_dataset.sr

valid_dataset=two_stage_LibriTTS(audio_dir="dev_full_file.json")
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=2,shuffle=True)

def spectral_reconstruction_loss(x, G_x):
    s=[2**i for i in range(6,12)]
    hop=[2**i//4 for i in range(6,12)]
    if args.rec_type == 1:
        stftloss=MultiResolutionSTFTLoss(fft_sizes=s,hop_sizes=hop,win_lengths=s,factor_sc=1, factor_mag=1).to(device)
        loss=stftloss(G_x.squeeze(1),x.squeeze(1))
    else:
        loss=MultiResolutionLoss(x, G_x, sr, device).to(device)
    return loss

def discriminator_loss(wave_disc_x,wave_disc_G_x,stft_disc_x,stft_disc_G_x):
    real_loss_d=0.0
    generated_loss_d=0.0
    for wave_disc in wave_disc_x:
        real_loss_d+=F.relu(1-wave_disc).mean()
    real_loss_d+=F.relu(1-stft_disc_x).mean()
    real_loss_d/=4

    for wave_disc in wave_disc_G_x:
        generated_loss_d+=F.relu(1+wave_disc).mean()
    generated_loss_d+=F.relu(1+stft_disc_G_x).mean()
    generated_loss_d/=4
    return real_loss_d,generated_loss_d

def generator_loss(wave_disc_G_x,stft_disc_G_x):
    adv_g_loss=0.0
    for wave_disc in wave_disc_G_x:
        adv_g_loss+=-wave_disc.mean()
    adv_g_loss+=-stft_disc_G_x.mean()
    adv_g_loss/=4
    return adv_g_loss

def feature_loss(features_wave_disc_x,features_wave_disc_G_x,features_stft_disc_x,features_stft_disc_G_x):
    feat_loss=0.0
    for i in range(3):
      for j in range(int(len(features_wave_disc_x)/3-1)):
        feat_loss+=F.l1_loss(features_wave_disc_G_x[i*7+j],features_wave_disc_x[i*7+j].detach())    
    for i in range(len(features_stft_disc_x)-1):
        feat_loss+=F.l1_loss(features_stft_disc_G_x[i],features_stft_disc_x[i].detach())
    feat_loss=feat_loss/(len(features_wave_disc_x)-3+len(features_stft_disc_x)-1)
    return feat_loss


def train_stage_1(epoch):
    encoder.train()
    vq.train()
    decoder.train()

    print("training epoch:",epoch)
    train_real_loss_d = 0.0
    train_generated_loss_d = 0.0
    train_loss_d=0.0
    train_adv_loss_g=0.0
    train_feat_loss_g=0.0
    train_rec_loss_g=0.0
    train_loss_g=0.0
    train_vq_loss_g=0.0
    for tup in tqdm(train_loader):
        x=tup[0]
        target=tup[1]
        target=target.to(device)
        x = x.to(device)
        e = encoder(x) 
        q,commit_loss=vq(e.permute(0,2,1),args.target_bit)
        e=q.permute(0,2,1)
        G_x = decoder(e)
        train_vq_loss_g+=commit_loss.item()

        rec_loss=spectral_reconstruction_loss(target,G_x)
        loss_g=rec_loss
        train_rec_loss_g+=rec_loss.item()
        train_loss_g += loss_g.item()
        encoder.zero_grad() 
        vq.zero_grad() 
        decoder.zero_grad() 
        loss_g.backward()
        optimizer_g.step()
        #updata tensorboard
        writer.add_scalar('train_d',train_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_g',train_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_real_d',train_real_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_generated_d',train_generated_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_adv_g',train_adv_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_feat_g',train_feat_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_rec_g',train_rec_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_vq_g',train_vq_loss_g/len(train_loader),epoch)

    print("train loss(d/g):",train_loss_d/len(train_loader),train_loss_g/len(train_loader))


def train_stage_2(epoch):
    for p in encoder.parameters():
        p.requires_grad=False
    for p in vq.parameters():
        p.requires_grad=False
    for p in decoder_d.parameters():
        p.requires_grad=False
    decoder.train()
    wave_disc.train()
    stft_disc.train()

    print("training epoch:",epoch)
    train_real_loss_d = 0.0
    train_generated_loss_d = 0.0
    train_loss_d=0.0
    train_adv_loss_g=0.0
    train_feat_loss_g=0.0
    train_rec_loss_g=0.0
    train_loss_g=0.0
    train_vq_loss_g=0.0
    for tup in tqdm(train_loader):
        x=tup[0]
        target=tup[1]
        target=target.to(device)
        x = x.to(device)
        e = encoder(x)
        q,commit_loss=vq(e.permute(0,2,1),args.target_bit)
        e=q.permute(0,2,1)
        G_x = decoder(e)
        train_vq_loss_g+=commit_loss.item()
        #Train Discriminator
        wave_disc_x, features_wave_disc_x,_ = wave_disc(target)
        wave_disc_G_x, _,_ = wave_disc(G_x.detach())
        x_stft = torch.stft(target.squeeze(1),n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH,window=getattr(torch, "hann_window")(WINDOW_LENGTH).to(device))
        G_x_stft = torch.stft(G_x.squeeze(1),n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH,window=getattr(torch, "hann_window")(WINDOW_LENGTH).to(device))
        stft_disc_x,features_stft_disc_x=stft_disc(x_stft.permute(0,3,2,1))
        stft_disc_G_x,_=stft_disc(G_x_stft.permute(0,3,2,1).detach())
        real_loss_d,generated_loss_d= discriminator_loss(wave_disc_x,wave_disc_G_x,stft_disc_x,stft_disc_G_x)
        loss_d = real_loss_d+generated_loss_d
        train_real_loss_d+=real_loss_d.item()
        train_generated_loss_d+=generated_loss_d.item()
        train_loss_d+=loss_d.item()
        stft_disc.zero_grad()
        wave_disc.zero_grad()            
        loss_d.backward()
        optimizer_d.step()

        #Train Generator 
        #reconstruction loss
        rec_loss=spectral_reconstruction_loss(target,G_x)
        loss_g=args.lambda_rec*rec_loss
        train_rec_loss_g+=rec_loss.item()

        wave_disc_G_x, features_wave_disc_G_x,_ = wave_disc(G_x)
        stft_disc_G_x,features_stft_disc_G_x=stft_disc(G_x_stft.permute(0,3,2,1))
        #adv loss
        adv_g_loss=generator_loss(wave_disc_G_x,stft_disc_G_x)
        train_adv_loss_g+=adv_g_loss.item()
        loss_g+=args.lambda_adv*adv_g_loss
        #feature loss
        feat_loss=feature_loss(features_wave_disc_x,features_wave_disc_G_x,features_stft_disc_x,features_stft_disc_G_x)
        train_feat_loss_g+=feat_loss.item()
        loss_g+=args.lambda_feat*feat_loss
        train_loss_g += loss_g.item()
        encoder.zero_grad() 
        vq.zero_grad() 
        decoder.zero_grad() 
        loss_g.backward()
        optimizer_g.step()
        #updata tensorboard
        writer.add_scalar('train_d',train_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_g',train_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_real_d',train_real_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_generated_d',train_generated_loss_d/len(train_loader),epoch)
        writer.add_scalar('train_adv_g',train_adv_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_feat_g',train_feat_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_rec_g',train_rec_loss_g/len(train_loader),epoch)
        writer.add_scalar('train_vq_g',train_vq_loss_g/len(train_loader),epoch)

    print("train loss(d/g):",train_loss_d/len(train_loader),train_loss_g/len(train_loader))


def eval(epoch):
    with torch.no_grad():
        encoder.eval()
        vq.eval()
        decoder.eval()
        if args.stage==2:  
            wave_disc.eval()
            stft_disc.eval()  
        valid_loss_g = 0.0
        valid_loss_d = 0.0
        for tup in tqdm(valid_loader):
            x=tup[0]
            target=tup[1]
            target=target.to(device)
            x = x.to(device)
            e = encoder(x) 
            q,commit_loss=vq(e.permute(0,2,1),args.target_bit)
            e=q.permute(0,2,1)
            G_x = decoder(e)
            #reconstruction loss
            rec_loss=spectral_reconstruction_loss(target,G_x)
            valid_loss_g+=args.lambda_rec*rec_loss.item()
            if args.stage==2:
                wave_disc_x, features_wave_disc_x,_ = wave_disc(target)
                wave_disc_G_x, features_wave_disc_G_x,_ = wave_disc(G_x)
                x_stft = torch.stft(target.squeeze(1),n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH,window=getattr(torch, "hann_window")(WINDOW_LENGTH).to(device))
                G_x_stft = torch.stft(G_x.squeeze(1),n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH,window=getattr(torch, "hann_window")(WINDOW_LENGTH).to(device))
                stft_disc_x,features_stft_disc_x=stft_disc(x_stft.permute(0,3,2,1))
                stft_disc_G_x,features_stft_disc_G_x=stft_disc(G_x_stft.permute(0,3,2,1))
                #adv loss
                adv_g_loss=generator_loss(wave_disc_G_x,stft_disc_G_x)
                valid_loss_g+=args.lambda_adv*adv_g_loss.item()
                #feature loss
                feat_loss=feature_loss(features_wave_disc_x,features_wave_disc_G_x,features_stft_disc_x,features_stft_disc_G_x)
                valid_loss_g+=args.lambda_feat*feat_loss.item()

                #dicriminator loss
                real_loss_d,generated_loss_d= discriminator_loss(wave_disc_x,wave_disc_G_x,stft_disc_x,stft_disc_G_x)
                valid_loss_d=valid_loss_d+real_loss_d.item()+generated_loss_d.item()
                
        
        writer.add_scalar('valid_d',valid_loss_d/len(valid_loader),epoch)
        writer.add_scalar('valid_g',valid_loss_g/len(valid_loader),epoch)
        print("valid loss(d/g):",valid_loss_d/len(valid_loader),valid_loss_g/len(valid_loader))


if __name__ == "__main__":
    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        if args.stage==1:
            train_stage_1(epoch)
            eval(epoch)
            if (epoch+1)%50==0:
                torch.save(encoder.state_dict().copy(),os.path.join(args.output_path,str(epoch)+"_encoder.pth"))
                torch.save(vq.state_dict().copy(),os.path.join(args.output_path,str(epoch)+"_vq.pth"))
                torch.save(decoder.state_dict().copy(),os.path.join(args.output_path,str(epoch)+"_decoder.pth"))

        elif args.stage==2:
            train_stage_2(epoch)
            eval(epoch)
            if (epoch+1)%50==0:
                torch.save(decoder.state_dict().copy(),os.path.join(args.output_path,str(epoch)+"_decoder.pth"))
                torch.save(stft_disc.state_dict().copy(),os.path.join(args.output_path,"final_stft.pth"))
                torch.save(wave_disc.state_dict().copy(),os.path.join(args.output_path,"final_wave.pth"))

    writer.close()    
    os.system("shutdown")