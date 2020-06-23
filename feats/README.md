# How to extract feats

## 1: c3d features of videos

### make a input list

`input` is a text containing a list of target videos. You can make one by:

```
cd /path/to/video
ls -R *.mp4 > input
```

### extract c3d features

We use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

```
python3 main.py --input ./input --video_root path/to/video --output ./output.json --model resnet-101-kinetics.pth --mode feature --model_name resnet --model_depth 101 --resnet_shortcut B --batch_size 16
# on our 8GB 2070super, the max batch size is 16
```

### turn json into npy

use `c3djson_to_npy.py`

## 2: MFCC features of sound

Use `mfcc_feats.py` to do that.

## 3: openpose features of sign language

Use `openpose_feats.py` to do that. You need to have [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) installed.

## 4: preprocess

`prepro_feats.py`, `prepro_vocab.py`
