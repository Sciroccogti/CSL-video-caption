import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from cv2 import cv2 as cv
from tqdm import tqdm

# C, H, W = 3, 224, 224


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_frames(video, dst, overwrite):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            if overwrite:
                logging.info("cleanup: " + dst + "/")
                shutil.rmtree(dst)
            else:
                logging.info("skipping: " + dst + "/")
                return
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=114:144",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, onWrapper):
    # global C, H, W

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    logging.info("save video feats to %s" % (dir_fc))

    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    video_list += glob.glob(os.path.join(params['video_path'], '*.avi'))
    pbar = tqdm(video_list)
    for video in pbar:
        video_id = video.split("/")[-1].split(".")[0]
        pbar.set_description(video_id)
        dst = dir_fc + '/' + video_id
        extract_frames(video, dst, params['overwrite'])

        outfile = os.path.join(dir_fc, video_id + '.npy')
        # print(os.path.exists(outfile))
        # print(not params['overwrite'])
        # print(os.path.exists(outfile) and (not params['overwrite']))
        if os.path.exists(outfile) and (not params['overwrite']):
            logging.info("skipping %s" % (outfile))
            continue

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        # 124 = 12 + 70 + 21*2, 3: x, y, confidence
        video_feats = np.zeros((len(image_list), 124, 3))
        i = 0

        for imagePath in image_list:
            # print(imagePath)
            try:
                datum = op.Datum()
                imageToProcess = cv.imread(imagePath)
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                if not params['no_display']:
                    cv.imshow(dst, datum.cvOutputData)

                # 12 points for upbody
                upbody = np.concatenate(
                    (datum.poseKeypoints[0, 0:8, :3], datum.poseKeypoints[0, 15:19, :3]))
                # print(datum.faceKeypoints.shape)
                video_feats[i] = np.concatenate(
                    (upbody, datum.faceKeypoints[0], datum.handKeypoints[0][0], datum.handKeypoints[1][0]))
            except Exception as e:
                print(imagePath + ':\n' + str(e))
                logging.error(imagePath + ':\n' + str(e))
                break
            i += 1
        # Save the inception features
        np.save(outfile, video_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/openpose', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')
    parser.add_argument("--no_display", default=True, type=str2bool,
                        help="Enable to disable the visual display.")
    parser.add_argument("--model_folder", type=str, default="./models/",
                        help="path to openpose models")
    parser.add_argument("--overwrite", dest='overwrite', type=str2bool, default=True,
                        help="Enable to skip existed files")

    # import openpose
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    args = parser.parse_args()
    params = vars(args)
    opParams = params.copy()
    opParams.pop('output_dir')
    opParams.pop('n_frame_steps')
    opParams.pop('video_path')
    opParams.pop('no_display')
    opParams.pop('overwrite')
    opParams["face"] = True
    opParams["hand"] = True

    opWrapper = op.WrapperPython()
    opWrapper.configure(opParams)
    opWrapper.start()

    starttime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.basicConfig(filename=starttime + '.log', level=logging.INFO)

    extract_feats(params, opWrapper)

