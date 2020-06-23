import argparse
import glob
import os
import sys

import audioread
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm


def mp3tomfcc(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    return mfcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str,
                        default='data/audio', help='path to audio dataset')
    parser.add_argument("--output_dir", type=str, default='data/feats/mfcc',
                        help='directory to store mfcc features')
    args = parser.parse_args()
    params = vars(args)

    if not os.path.isdir(params['output_dir']):
        os.makedirs(params['output_dir'])

    audio_list = glob.glob(os.path.join(params['audio_dir'], '*.wav'))
    audio_list += glob.glob(os.path.join(params['audio_dir'], '*.aac'))
    pbar = tqdm(audio_list)
    frame = []

    for audio in pbar:
        audio_id = audio.split("/")[-1].split(".")[0]
        pbar.set_description(audio_id)

        feat = mp3tomfcc(audio)
        frame.append(feat.shape[1])

        np.save(params['output_dir'] + '/mfcc_' + audio_id + '.npy', feat)


    print('median frame num:', np.median(frame))
    print('average frame num:', np.mean(frame))
    print('minimum frame num:', np.min(frame))
    print('maximum frame num:', np.max(frame))
