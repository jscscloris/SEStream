# SEStream
This is the code for the paper "A Two-Stage Training Framework for Joint Speech Compression and Enhancement". 

Some results are provided in the demo-audio folder.

The code is modified from wesbz' code at: https://github.com/wesbz/SoundStream

## Datasets

We use LibriTTS train-clean-100 and train-clean-360 as our dataset. The noise dataset is synthesized by train-clean-100\dev-clean and Freesound. The pipline of dataset is shown as followed:
```
./generate_data.py
./train.py
./data/
/LibriTTS/
├── train-clean-100/
├── train-clean-360/
└── dev-clean/
  ├── 84/
  ├── ...
  └── 8842/
    ├── 302196/       
    ├── ...
    └── 304647/
      ├── 8842_304647_000005_000000.wav
        ...
/noise/
├── train-clean-100/
└── dev-clean/
  ├── 84_121123_000007_000001.wav
    ...
```

## Getting started

### Python requirements

This code requires:

- Python 3.8
- torch 1.11.0
- torchaudio 0.11.0

### Preparing training dataset

To generate the json file for training:
```
python generate_data.py --dataset "./data/LibriTTS/" --noise_dataset "./data/noise/"
```

### Training networks

To train the first stage of SEStream with quantizer dropout:
```
python train.py --stage 1 --num_gpu 1 --target_bit 0 
```
Then train the second stage of SEStream quantizer dropout:
```
python train.py --stage 2 --num_gpu 1 --target_bit 0 
```

### Evaluation methods

The evaluation is based on visqol as shown in https://github.com/google/visqol.

