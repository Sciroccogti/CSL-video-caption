from cv2 import cv2 as cv
import argparse
import glob
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir")
args = parser.parse_args()
params = vars(args)

videoList = glob.glob(os.path.join(params['video_dir'], '*.mp4'))
videoList += glob.glob(os.path.join(params['video_dir'], '*.avi'))
videoList.sort()
frame = []
duration = []

for video in videoList:
    cap = cv.VideoCapture(video)
    frame.append(cap.get(7)) # frame num
    duration.append(cap.get(7) / cap.get(5)) # duration in sec

print('median frame num:', np.median(frame))
print('average frame num:', np.mean(frame))
print('minimum frame num:', np.min(frame))
print('maximum frame num:', np.max(frame))

print('\nmedian duration:', np.median(duration), 's')
print('average duration num:', np.mean(duration), 's')
print('minimum duration num:', np.min(duration), 's')
print('maximum duration num:', np.max(duration), 's')

