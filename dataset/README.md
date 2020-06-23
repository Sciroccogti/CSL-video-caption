# How to make a dataset

Below is an example of our origin dataset（720*576）:

![](../assets/G_00045.gif)

## 1: video

### normalize frame nums

we will extract c3d feats, which require the videos to have same length,
 or we say, same frame nums. You can use `resample.py` along with `frames_count.py` to do this:

```bash
python3 dataset/video_info.py --video_dir path/to/videos
# You will get the statistics characteristics of your videos' frame number.
# According them you can determine the 'target_frame' for resample.py.
# We recommend to set the target_frame to median frame num
python3 dataset/resample.py --video_dir path/to/videos --target_frame 224
```

### crop

We will crop the main sight of the origin video out as a new set called "video" (576*450):

```bash
ffmpeg -i in.avi -vf crop=576:450:144:0 out.avi # crop=width:height:x:y
# Also you can use crop.py
```

We got this:

![](../assets/G_video_00045.gif)

## 2: sound

We extract the sound of the videos by ffmpeg:
```bash
python3 dataset/audio_extract.py --video_dir path/to/videos --target_duration 5.261
```

We need to normalize the length of mfcc feats, so the extracted audios should have
 same durations. In our test, we want the length of mfcc feats be as long as those
 of video feats, which are 224. Thus we should set the target_duration as 5.261
 = (224 + 8) / 44.1, where `+8` ensures the length is larger than 224 and smaller
 than 224 + 16

## 3: hand

We will crop the sign language part out as a new set called "hand" (144*114): 

```bash
ffmpeg -i in.avi -vf crop=114:144:30:372 out.avi # crop=width:height:x:y
```

Then we get a "hand" video like this:

![](../assets/G_hand_00045.gif)

## others

### merge several datasets

If you want to merge several datasets and meets difficulty while dealing with their jsons,
 we have a script `merge_json.py` for you:

```
python3 dataset/merge_json.py --in_json data00400/English_caption\(400-590\).json data00000/english_full_caption_new_0-50.json --target_list sentences --out_json english_caption_0+400.json
```
