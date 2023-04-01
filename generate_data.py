import glob
import torchaudio
import argparse
from tqdm import tqdm
import json
parser=argparse.ArgumentParser(description="the generation of dataset")
parser.add_argument("--dataset",type=str,default="./data/LibriTTS/",help="the path for dataset")
parser.add_argument("--noise_dataset",type=str,default="./data/noise/",help="the path for noisy dataset")
args=parser.parse_args()



    
def generate_dataset_list(dataset_path,noisy_path):
    train_large_file=[]
    file_names=glob.glob(dataset_path + "train-clean-100/*/*/*.wav")
    for audio in tqdm(file_names):
        wav,sr=torchaudio.load(audio)
        if wav.shape[1]<8640:
            continue
        train_large_file.append({"length":wav.shape[1],"clean":audio,"input":audio})
        noisy= noisy_path + "train-clean-100/"+audio.split("/")[-1]
        train_large_file.append({"length":wav.shape[1],"clean":audio,"input":noisy})
    
    file_names=glob.glob(dataset_path + "train-clean-360/*/*/*.wav")
    for audio in tqdm(file_names):
        wav,sr=torchaudio.load(audio)
        if wav.shape[1]<8640:
            continue
        train_large_file.append({"length":wav.shape[1],"clean":audio,"input":audio})
    with open("train_full_file.json","w") as f:
        f.write(json.dumps(train_large_file))

    dev_large_file=[]
    file_names=glob.glob(dataset_path + "dev-clean/*/*/*.wav")
    for audio in tqdm(file_names):
        wav,sr=torchaudio.load(audio)
        if wav.shape[1]<8640:
            continue
        dev_large_file.append({"length":wav.shape[1],"clean":audio,"input":audio})
        noisy= noisy_path + "dev-clean/"+audio.split("/")[-1]
        dev_large_file.append({"length":wav.shape[1],"clean":audio,"input":noisy})
    
    with open("dev_full_file.json","w") as f:
        f.write(json.dumps(dev_large_file))

generate_dataset_list(args.dataset,args.noise_dataset)
