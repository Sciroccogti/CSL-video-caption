import json
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json", default='./output.json')
parser.add_argument("--output_dir", default='data/3dfeature/')
args = parser.parse_args()
params = vars(args)

videos = json.load(open(params['json'], 'r'))

if not os.path.exists(params['output_dir']):
    os.makedirs(params['output_dir'])

for video in videos:
    name = str(video["video"].split(".")[0])
    segments = video["clips"]
    feat = np.zeros((len(segments), 2048))
    # if len(segments) != 14:
    #     print(name, len(segments))
    i = 0
    for segment in segments:
        feat[i] = segment["features"]
        i += 1
    np.save(params['output_dir'] + name + ".npy", feat)
