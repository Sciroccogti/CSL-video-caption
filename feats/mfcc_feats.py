import argparse
import glob
import os
import sys
import wave

import audioread
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy
import scipy
from sklearn import preprocessing
from tqdm import tqdm

matplotlib.use('TkAgg')


def mfcc(audioPath, output_dir):
    '''
    :param audioPath: path to audio file
    :param output_dir: path to save the feats
    '''
    t, spe = librosa.load(audioPath)
    mfccs = librosa.feature.mfcc(t, sr=spe)
    audio_id = audio.split("/")[-1].split(".")[0]
    name = output_dir + '/mfcc_' + audio_id + ".npy"
    numpy.save(name, mfccs)


def decode2wav(filename, outname):
    f = audioread.audio_open(filename)
    nsample = 0
    for buf in f:
        nsample += 1
    f.close()
    with audioread.audio_open(filename) as f:
        print("input file: channels=%d, samplerate=%d, duration=%d" %
              (f.channels, f.samplerate, f.duration))
        channels = f.channels
        samplewidth = 2
        samplerate = f.samplerate
        compresstype = "NONE"
        compressname = "not compressed"
        outwav = wave.open(outname, 'wb')
        outwav.setparams((channels, samplewidth, samplerate,
                          nsample, compresstype, compressname))
        for buf in f:
            outwav.writeframes(buf)
        outwav.close()


def pcm2wav(srcname, outname, channels, samplewidth, samplerate):
    fs = os.path.getsize(srcname)
    nsample = fs / samplewidth
    outwav = wave.open(outname, 'wb')
    outwav.setparams((channels, samplewidth, samplerate,
                      nsample, "NONE", "not cmopressed"))
    fsrc = open(srcname, 'rb')
    outwav.writeframes(fsrc.read())
    fsrc.close()
    outwav.close()


def Normalize(data):
    m = numpy.mean(data)
    mx = numpy.max(data)
    mn = numpy.min(data)
    return [[(float(data[i][j]) - m) / (mx - mn) for i in range(data.shape[0])]for j in range(data.shape[1])]


def reps(data):
    length = numpy.size(data)
    # print(length)
    if length % 2048 != 0:
        a = numpy.zeros((1, 2048-length % 2048))
        #print(numpy.shape(data), numpy.shape(a))
        data = numpy.concatenate((data, a), axis=1)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str,
                        default='data/audio', help='path to audio dataset')
    parser.add_argument("--c3d_path", type=str,
                        default='data/feats/c3d_feats', help='path to c3d feats')
    parser.add_argument("--mfcc_dir", type=str,
                        default='data/feats/mfcc', help='directory to store mfcc feats')
    parser.add_argument("--concat_dir", type=str,
                        default='data/feats/concat', help='directory to store mfcc+c3d features')

    args = parser.parse_args()
    params = vars(args)

    if not os.path.isdir(params['mfcc_dir']):
        os.mkdir(params['mfcc_dir'])

    if not os.path.isdir(params['concat_dir']):
        os.mkdir(params['concat_dir'])

    audio_list = glob.glob(os.path.join(params['audio_path'], '*.wav'))
    audio_list += glob.glob(os.path.join(params['audio_path'], '*.aac'))
    pbar = tqdm(audio_list)

    for audio in pbar:
        audio_id = audio.split("/")[-1].split(".")[0]
        pbar.set_description(audio_id)

        mfcc(audio, params['mfcc_dir'])

        filename = params['mfcc_dir'] + 'mfcc_' + audio_id + ".npy"
        c3d_feats = params['c3d_path'] + audio_id + ".npy"
        try:
            voice = numpy.load(filename)
            voice = Normalize(voice)
            voice = numpy.array(voice)
            video = numpy.load(c3d_feats)
            # print(numpy.shape(voice),numpy.shape(video))
            voice = voice.reshape(1, -1)
            voice = reps(voice)
            voice = voice.reshape(-1, 2048)
            # print(numpy.shape(voice),numpy.shape(video))
            voice = numpy.concatenate((voice, video))
            # print(numpy.shape(voice))
            numpy.save(params['concat_dir'] + '/concat_' +
                       audio_id + ".npy", voice)
        except Exception as err:
            print(audio_id, err)
