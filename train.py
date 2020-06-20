import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader
import visdom


def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    viz = visdom.Visdom(env='train')
    loss_win = viz.line(np.arange(1), opts={'title':'loss'})
    
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        # print(opt["self_crit_after"])
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]: #每多少次保存一下
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        # print(model)

        for data in loader:
            # print(data)
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            # voice_feats = data['voice_feats'].cuda()
            hand_feats = data['hand_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()
            #print(sc_flag)
            optimizer.zero_grad()
            if not sc_flag:
                # seq_probs, _ = model(fc_feats, voice_feats, hand_feats, labels, 'train')
                seq_probs, _ = model(fc_feats, hand_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            # todo 下面else部分没有修改声音和手语的内容
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("?iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss))
                viz.line(Y=np.array([train_loss]), X=np.array([epoch]), win=loss_win, update='append')
            else:
                print("??iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch, np.mean(reward[:, 0])))

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            # print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        # # 声音encoder
        # encoder_voice = EncoderRNN(
        #     opt["dim_voice"],
        #     opt["dim_hidden"],
        #     bidirectional=opt["bidirectional"],
        #     input_dropout_p=opt["input_dropout_p"],
        #     rnn_cell=opt['rnn_type'],
        #     rnn_dropout_p=opt["rnn_dropout_p"])
        # 手语encoder
        encoder_hand = EncoderRNN(
            opt["dim_hand"],
            opt["dim_hand_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])

        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        # model = S2VTAttModel(encoder, encoder_voice, encoder_hand, decoder)
        model = S2VTAttModel(encoder, encoder_hand, decoder)
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])
    # print(dataloader)
    # print(crit)
    # print(optimizer)

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    # print(opt)
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    # print('save opt details to %s' % (opt_json))
    main(opt)
