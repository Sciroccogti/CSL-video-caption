import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in', type=str,
                    help='msr_vtt videoinfo json')
parser.add_argument('--out', type=str,
                    help='caption json file')
args = parser.parse_args()
params = vars(args)  # convert to ordinary dict

in_caption = json.load(open(params['in'], 'r'))

sentences = {
    'sentences': []
}

for id in in_caption:
    i = 0
    for sentence in in_caption[id]['captions']:
        i += 1
        sentences['sentences'].append({
            'caption': sentence,
            'video_id': id,
            'sen_id': i
        })

json.dump(sentences, open(params['out'], 'w'))
