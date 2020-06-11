import argparse
import glob
import os
import shutil
import subprocess
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir")
parser.add_argument("--output_dir")
parser.add_argument("--mode", type=str, default='video', help='video or hand')
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
    video_id = video.split("/")[-1].split(".")[0]
    video_type = video.split("/")[-1].split(".")[1]
    pbar.set_description(video_id)
    if params['mode'] == 'video':
        ffmpeg_command = ['ffmpeg', '-y', '-i', video, '-an', '-vf', 'crop=576:450:144:0',
                      params['output_dir'] + video_id + '.' + video_type]
    elif params['mode'] == 'hand':
        ffmpeg_command = ['ffmpeg', '-y', '-i', video, '-an', '-vf', 'crop=114:144:30:372',
                      params['output_dir'] + 'hand' + video_id + '.' + video_type]
    with open(os.devnull, "w") as ffmpeg_log:
        subprocess.call(ffmpeg_command, stdout=ffmpeg_log,
                        stderr=ffmpeg_log)
