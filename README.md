# SEStream
The code for "A Two-Stage Training Framework for Joint Speech Compression and Enhancement"

The framework of encoder - decoder and RVQ relies on wesbz' repository. https://github.com/wesbz/SoundStream

## Datasets

In this section, we will introduce the datasets we use for our experiment. 

## Getting started

### Python requirements

This code requires:

- Python 3.8
- torch 1.11.0
- torchaudio 0.11.0

### Training networks

To train the first stage of SEStream:
```
python train.py --stage 1 --num_gpu 1 --target_bit 0 
```
Then train the second stage of SEStream:
```
python train.py --stage 2 --num_gpu 1 --target_bit 0 
```



