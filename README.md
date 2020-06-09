# CSL-video-caption

We want to fetch a better caption via inputting sound and sign language into the LSTM.

Based on the [previous work](https://github.com/Paymemoney/Video-Caption), which added sound to video caption.

## What is the dataset

### 1: video

 Below is an example of our origin dataset（720*576）:

![](assets/G_00045.gif)

### 2: sound

We extract the sound of the videos by ffmpeg.

### 3: hand
We will crop the sign language part out as a new set called "hand" （144*114）by the line: 

```bash
ffmpeg -i in.mp4 -vf crop=114:144:30:372 out.mp4 # crop=width:height:x:y
```

Then we get a "hand" video like this:

![](assets/G_hand_00045.gif)

## How to train

### 1: c3d features of videos

We use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

### 2: MFCC features of sound

We did this in [previous work](https://github.com/Paymemoney/Video-Caption), you can find the file `mfcc.py` there.

### 3: openpose features of sign language

We are working on this.
