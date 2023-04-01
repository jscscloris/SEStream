from torch.utils.data import Dataset
import torchaudio
import json
import torch
import random
import glob
import os
import numpy as np

SLICE=8640

class two_stage_LibriTTS(Dataset):
    def __init__(self,audio_dir):
        super().__init__()
        with open(audio_dir,"r") as file:
            str_d=file.read()
            data=json.loads(str_d)
        self.filenames=data 
        self.idx=0
        _, self.sr=torchaudio.load(self.filenames[0]["input"])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,index):
        self.idx=index
        leng=self.filenames[index]["length"]
        if leng-SLICE-1<=0:
            offset=0
        else:
            offset=random.randint(0,leng-SLICE-1)
        input_audio_name=self.filenames[index]["input"]
        clean_audio_name=self.filenames[index]["clean"]
        clean_audio=torchaudio.load(clean_audio_name,frame_offset=offset,num_frames=SLICE,normalize=True)[0]
        input_audio=torchaudio.load(input_audio_name,frame_offset=offset,num_frames=SLICE,normalize=True)[0]
        target=clean_audio
        max_ad=torch.max(torch.abs(input_audio))
        if max_ad!=0:
            rd=random.uniform(0.3,1.0)
            input2=input_audio/max_ad*0.95*rd
            target2=target/max_ad*0.95*rd
        else:
            input2=input_audio
            target2=target
        return input2,target2

    def get_name(self):
        return self.filenames[self.idx]["audio"]