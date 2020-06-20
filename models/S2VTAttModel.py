import torch.nn as nn
import torch


class S2VTAttModel(nn.Module):
    def __init__(self, encoder, encoder_hand, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        # self.encoder_voice = encoder_voice
        self.encoder_hand = encoder_hand
        self.decoder = decoder

    def forward(self, vid_feats, hand_feats, target_variable=None,
                mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs1, encoder_hidden1 = self.encoder(vid_feats)
        # todo 修改下面的vid_feats，变成声音和手语的特征,同时要传入参数
        # encoder_outputs2, encoder_hidden2 = self.encoder_voice(voice_feats)
        encoder_outputs3, encoder_hidden3 = self.encoder_hand(hand_feats)
        # todo 我不知道下面的应该是哪一维进行拼接，到时候输出测试一下
        # encoder_outputs = torch.cat([encoder_outputs1, encoder_outputs2, encoder_outputs3], 2)
        # encoder_hidden = torch.cat([encoder_hidden1, encoder_hidden2, encoder_hidden3], 2)
        print('encoder_outputs1', encoder_outputs1.shape)
        print('encoder_outputs3', encoder_outputs3.shape)
        encoder_outputs = torch.cat([encoder_outputs1, encoder_outputs3], 1)
        encoder_hidden = torch.cat([encoder_hidden1, encoder_hidden3], 1)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds
