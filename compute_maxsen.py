import re
import json
import argparse
import numpy as np


def main(params):
    
    videos = json.load(open(params['input_json'], 'r'))
    max = 0
    var = ''
    # print(type(videos))
    # print(type(videos['G_00400']))
    # print(type(videos['G_00400']['final_captions']))
    # print(type(videos['G_00400']['final_captions'][0]))
    # print(len(videos['G_00400']['final_captions'][0])-2)

    for i in videos:
        for j in videos[i]['final_captions']:
            if len(j)>max:
                max = len(j)
                var = str(i)
    print(max-2)
    print(var)
    
    # for ins in videos['G_00495']['final_captions']:
    #     print(ins)
    #     print(len(ins)-2)

    # video_caption = {}
    # for i in videos:
    #     if i['video_id'] not in video_caption.keys():
    #         video_caption[i['video_id']] = {'captions': []}
    #     video_caption[i['video_id']]['captions'].append(i['caption'])
    # # create the vocab
    # vocab = build_vocab(video_caption, params)
    # itow = {i + 2: w for i, w in enumerate(vocab)}
    # wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    # wtoi['<eos>'] = 0
    # itow[0] = '<eos>'
    # wtoi['<sos>'] = 1
    # itow[1] = '<sos>'

    # out = {}
    # out['ix_to_word'] = itow
    # out['word_to_ix'] = wtoi
    # out['videos'] = {'train': [], 'val': [], 'test': []}
    # videos = json.load(open(params['input2_json'], 'r'))['videos']
    # for i in videos:
    #     out['videos'][i['split']].append(int(i['id']))
    # json.dump(out, open(params['info_json'], 'w'))
    # json.dump(video_caption, open(params['caption_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='data/caption.json',
                        help='msr_vtt videoinfo json')

    args = parser.parse_args()
    params = vars(args) 
    main(params)