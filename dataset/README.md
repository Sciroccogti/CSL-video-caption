# How to make a dataset

Below is an example of our origin dataset（720*576）:

![](../assets/G_00045.gif)

## 1: video

We will crop the main sight of the origin video out as a new set called "video" (576*450):

```bash
ffmpeg -i in.avi -vf crop=576:450:144:0 out.avi # crop=width:height:x:y
```

We got this:

![](../assets/G_video_00045.gif)

## 2: sound

We extract the sound of the videos by ffmpeg.

## 3: hand
We will crop the sign language part out as a new set called "hand" (144*114): 

```bash
ffmpeg -i in.avi -vf crop=114:144:30:372 out.avi # crop=width:height:x:y
```

Then we get a "hand" video like this:

![](../assets/G_hand_00045.gif)