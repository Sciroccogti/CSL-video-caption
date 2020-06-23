'''
This script helps you to merge several datasets' jsons into one.
'''

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--in_json', type=str, nargs='+',
                    help='path to jsons that need to be merged')
parser.add_argument('--target_list', type=str,
                    help='which list in the jsons should be merged')
parser.add_argument('--out_json', type=str,
                    help='output json')
args = parser.parse_args()
params = vars(args)

if not params['target_list']:
    print('Please set target_list !')
    exit(-1)

out_list = []
for i in params['in_json']:
    out_list += json.load(open(i))[params['target_list']]

out = {
    params['target_list']: out_list
}

json.dump(out, open(params['out_json'], 'w'))
