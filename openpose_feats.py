import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from cv2 import cv2
from tqdm import tqdm

# C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
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


def extract_feats(params):
    # global C, H, W
    # model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    video_list = video_list + glob.glob(os.path.join(params['video_path'], '*.avi'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = dir_fc + '/' + video_id
        extract_frames(video, dst)

        # image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        # samples = np.round(np.linspace(
        #     0, len(image_list) - 1, params['n_frame_steps']))
        # image_list = [image_list[int(sample)] for sample in samples]
        # images = torch.zeros((len(image_list), C, H, W))
        # for iImg in range(len(image_list)):
        #     img = load_image_fn(image_list[iImg])
        #     images[iImg] = img
        # with torch.no_grad():
        #     fc_feats = model(images.cuda()).squeeze()
        # img_feats = fc_feats.cpu().numpy()
        # # Save the inception features
        # outfile = os.path.join(dir_fc, video_id + '.npy')
        # np.save(outfile, img_feats)
        # # cleanup
        # shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/openpose', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')
    parser.add_argument("--no_display", default=False,
                        help="Enable to disable the visual display.")

    # import openpose
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    args = parser.parse_args()
    params = vars(args)
    # params = dict()
    params["model_folder"] = "models/"

    # pass other args to openpose?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1])-1: next_item = args[1][i+1]
    #     else: next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params: params[key] = next_item

    extract_feats(params)
    # # Starting OpenPose
    # opWrapper = op.WrapperPython()
    # opWrapper.configure(params)
    # opWrapper.start()

    # # Read frames on directory
    # imagePaths = op.get_images_on_directory(args[0].image_dir)
    # start = time.time()