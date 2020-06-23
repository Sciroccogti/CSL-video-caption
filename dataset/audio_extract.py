'''
extract the audios from videos and rescale them to make their duration as same.
'''

import argparse
import glob
import os
import shutil
import subprocess

import librosa
import numpy as np
from cv2 import cv2 as cv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir")
parser.add_argument("--target_duration", type=float,
                    default=5.261, help='usually duration = frame / 44.1')
parser.add_argument("--output_dir", type=str, default='data/audio/')
args = parser.parse_args()
params = vars(args)

videoList = glob.glob(os.path.join(params['video_dir'], '*.mp4'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.avi'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.wav'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.aac'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.m4a'))
videoList.sort()
pbar = tqdm(videoList)

if not os.path.exists(params['output_dir']):
    os.makedirs(params['output_dir'])

if not params['output_dir'].endswith('/'):
    params['output_dir'] += '/'

for video in pbar:
    video_id = video.split("/")[-1].split(".")[0]
    video_type = video.split("/")[-1].split(".")[1]
    pbar.set_description(video_id)

    if video_type == 'mp4' or video_type =='avi':
        cap = cv.VideoCapture(video)
        frame = cap.get(7)
        if frame == 0:
            print(video_id, 'has no frame or has an error with the video')
            continue
        duration = cap.get(7) / cap.get(5)
    else:
        y, _ = librosa.load(video)
        duration = librosa.get_duration(y)
        if duration == 0:
            print(video_id, 'has no frame or has an error with the audio')
            continue
    scale = duration / params['target_duration']
    tempo = ''

    while scale > 2.0 or scale < 0.5:
        if scale > 2.0:
            tempo += 'atempo=2.0,'
            scale /= 2.0
        else:
            tempo += 'atempo=0.5,'
            scale /= 0.5

    tempo += 'atempo=%f' % scale

    ffmpeg_command = ['ffmpeg', '-y', '-i', video, '-filter:a', tempo,
                        params['output_dir'] + video_id + '.aac']
    with open(os.devnull, "w") as ffmpeg_log:
        subprocess.call(ffmpeg_command, stdout=ffmpeg_log,
                        stderr=ffmpeg_log)
