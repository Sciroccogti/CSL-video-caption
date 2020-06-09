# CSL-video-caption

We want to fetch a better caption via inputting sound and sign language into the LSTM.

Based on the [previous work](https://github.com/Paymemoney/Video-Caption), which added sound to video caption.

## What is the dataset

see [How to make a dataset](dataset/README.md)

## How to train

### 1: c3d features of videos

We use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

### 2: MFCC features of sound

We did this in [previous work](https://github.com/Paymemoney/Video-Caption), you can find the file `mfcc.py` there.

### 3: openpose features of sign language

We are working on this.
