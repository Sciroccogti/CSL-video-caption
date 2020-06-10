'''
resample the videos to make them have same frame num
you can use `frames_count.py` to get the median frame num as "target frame"
'''

import argparse
import glob
import os
import shutil
import subprocess
from tqdm import tqdm

import numpy as np
from cv2 import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir")
parser.add_argument("--target_frame", type=int, default=224)
parser.add_argument("--output_dir", type=str, default='data/resampled/')
args = parser.parse_args()
params = vars(args)

videoList = glob.glob(os.path.join(params['video_dir'], '*.mp4'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.avi'))
videoList.sort()
pbar = tqdm(videoList)

if not os.path.exists(params['output_dir']):
    os.makedirs(params['output_dir'])

if not params['output_dir'].endswith('/'):
    params['output_dir'] += '/'

for video in pbar:
    cap = cv.VideoCapture(video)
    frame = cap.get(7)
    video_id = video.split("/")[-1].split(".")[0]
    video_type = video.split("/")[-1].split(".")[1]
    pbar.set_description(video_id)
    if frame == 0:
        print(video_id, 'has no frame or has an error with the video')
        continue

    if frame - params['target_frame'] < 8 and frame > params['target_frame']:
        try:
            shutil.copyfile(
                video, params['output_dir'] + video_id + video_type)
        except Exception as err:
            print(video_id, err)
    else:
        fps = cap.get(5)
        targetfps = str(fps * (params['target_frame'] + 8) / frame)
        # c3d extract one feature per 16 frames, plus 8 to make a median
        ffmpeg_command = ['ffmpeg', '-y', '-i', video, '-r', targetfps,
                          params['output_dir'] + video_id + '.' + video_type]
        with open(os.devnull, "w") as ffmpeg_log:
            subprocess.call(ffmpeg_command, stdout=ffmpeg_log,
                            stderr=ffmpeg_log)
