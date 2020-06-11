# How to extract feats

## 1: c3d features of videos

We use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

```
python3 main.py --input ./input --video_root path/to/video --output ./output.json --model resnet-101-kinetics.pth --mode feature --model_name resnet --model_depth 101 --resnet_shortcut B --batch_size 16
# on our 8GB 2070super, the max batch size is 16
```

`input` is a text containing a list of target videos. You can make one by:

```
cd /path/to/video
ls -R *.mp4 > input
```

## 2: MFCC features of sound

We did this in [previous work](https://github.com/Paymemoney/Video-Caption), you can find the file `mfcc.py` there.

## 3: openpose features of sign language

Use `openpose_feats.py` to do that. You need to have [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) installed.
