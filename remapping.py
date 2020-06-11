import argparse
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--errors_json',
    type=str,
    default='data/removedList.json',  # 储存的是有问题的视频编号
    help='path to the json file containing error video')
parser.add_argument(
    '--feats_dir',
    nargs='*',
    type=str,
    default='data/feats/c3d_feats/',
    help='path to the directory containing the preprocessed fc feats')
args = parser.parse_args()
params = vars(args)

error_list = json.load(open(params["errors_json"]))["errors"]
feats_list = glob.glob(os.path.join(params['feats_dir'], '*.npy'))
f = {}
idx = 0

for feat in feats_list:
    feat_id = feat.split("/")[-1].split(".")[0]
    if not feat_id in error_list:
        f[idx] = feat_id
        idx += 1

j = json.dumps(f)
f2 = open('mapping.json', 'w')
f2.write(j)
f2.close()
