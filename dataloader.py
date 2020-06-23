import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        self.with_hand = opt['with_hand']
        self.with_voice = opt['with_voice']
        self.hand_feats_dir = opt['hand_feats_dir']
        self.voice_feats_dir = opt['voice_feats_dir']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        id = self.splits[self.mode][ix]
        
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'G_%05i.npy' % (id))))
        fc_feat = np.concatenate(fc_feat, axis=1)

        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'G_%05i.npy'%(id)))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        
        if self.with_hand == 1:
            hand_feat = np.load(os.path.join(self.hand_feats_dir, 'handG_%05i.npy'%(id)))
            hand_pro = hand_feat[:224, :, 2]
            hand_pro = np.tile(hand_pro, (1, 2))
            hand_feat = np.reshape(hand_feat[:224, :, :2], (224, 248))

        if self.with_voice == 1:
            voice_feats = np.load(os.path.join(self.voice_feats_dir, 'mfcc_G_%05i.npy'%(id)))
            voice_feats = voice_feats[:60, :224].T
        # hand_feat = np.mean(hand_feat, axis=0, keepdims=True)

        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions['G_%05i'%(id)]['final_captions']
        gts = np.zeros((9, self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                if i >= 9:
                    break
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, 8)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        if self.with_hand == 1:
            data['hand_feats'] = torch.from_numpy(hand_feat).type(torch.FloatTensor)
            data['hand_pro'] = torch.from_numpy(hand_pro).type(torch.FloatTensor)
        if self.with_voice == 1:
            data['voice_feats'] = torch.from_numpy(voice_feats).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'G_%05i'%(id)
        return data

    def __len__(self):
        return len(self.splits[self.mode])
