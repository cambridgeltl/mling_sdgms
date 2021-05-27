# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
training SEMI-CLDC
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import math
import random
from random import shuffle
import torch
import torch.optim as optim
import numpy as np

from nn_model.xlsemicldc_model import XLSEMICLDCModel
from nn_model.aux_xlsemicldc_model import AUXXLSEMICLDCModel
from utils.early_stopping import EarlyStopping
import utils.ios
from utils.utils import enumerate_discrete
from tensorboardX import SummaryWriter
from train_cldc import get_data, get_lang_data, test, tsne2d
from train_semicldc import gen_task_info, get_batch, get_rest_batch, get_classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main(params, vocabs, model_dict, aux = False):
  # get data
  datas = get_data(params, vocabs)
  # get model
  if aux:
    m = AUXXLSEMICLDCModel(params, datas, model_dict = model_dict)
  else:
    m = XLSEMICLDCModel(params, datas, model_dict = model_dict)
  gen_xltask_info(params, m, datas)
  train(params, m, datas)


def train(params, m, datas):
  # early stopping
  es = EarlyStopping(mode = 'max', patience = params.cldc_patience)
  # set optimizer
  optimizer, dis_optimizer, dis_enc_optimizer = get_optimizer(params, m)

  # source language data
  src_lang, src_data = get_lang_data(params, datas, training = True)
  # target language data
  trg_lang, trg_data = get_lang_data(params, datas)

  assert(src_data.train_size == trg_data.train_size or trg_data.train_size == 0 or src_data.train_size == 0)
  # source label bs
  src_bs = params.cldc_bs
  if src_data.train_size > 0:
    n_batch = src_data.train_size // src_bs if src_data.train_size % src_bs == 0 else src_data.train_size // src_bs + 1
  elif src_data.train_size == 0:
    n_batch = trg_data.train_size // src_bs if trg_data.train_size % src_bs == 0 else trg_data.train_size // src_bs + 1

  # get the same n_batch for unlabelled data as well
  # batch size for unlabelled data
  src_rest_bs = src_data.rest_train_size // n_batch
  # get the same n_batch for targeted language training data
  trg_bs = trg_data.train_size // n_batch
  trg_rest_bs = trg_data.rest_train_size // n_batch

  # data index per category
  src_data_idxs = [list(range(len(train_idx))) for train_idx in src_data.train_idxs]
  src_rest_data_idxs = list(range(len(src_data.rest_train_idxs)))
  trg_data_idxs = [list(range(len(train_idx))) for train_idx in trg_data.train_idxs]
  trg_rest_data_idxs = list(range(len(trg_data.rest_train_idxs)))

  # number of iterations
  cur_it = 0
  # write to tensorboard
  writer = SummaryWriter('./history/{}'.format(params.log_path)) if params.write_tfboard else None
  # best dev/test
  bdev = 0
  btest = 0
  # current dev/test
  cdev = 0
  ctest = 0
  dev_class_acc = {}
  test_class_acc = {}
  # early stopping warm up flag, start es after train loss below some threshold
  es_flag = False
  # set io function
  out_semicldc = getattr(utils.ios, 'out_xlsemicldc_{}'.format(params.cldc_train_mode))

  for i in range(params.cldc_ep):
    for src_data_idx in src_data_idxs:
      shuffle(src_data_idx)
    shuffle(src_rest_data_idxs)
    for trg_data_idx in trg_data_idxs:
      shuffle(trg_data_idx)
    shuffle(trg_rest_data_idxs)
    for j in range(n_batch):
      src_train_idxs = []
      trg_train_idxs = []
      # iterate over labels
      for k, (src_data_idx, trg_data_idx) in enumerate(zip(src_data_idxs, trg_data_idxs)):
        if j < n_batch - 1:
          src_train_idxs.append(src_data_idx[int(j * src_bs * src_data.train_prop[k]): int((j + 1) * src_bs * src_data.train_prop[k])])
          src_rest_train_idxs = src_rest_data_idxs[j * src_rest_bs: (j + 1) * src_rest_bs]
          trg_train_idxs.append(trg_data_idx[int(j * trg_bs * trg_data.train_prop[k]): int((j + 1) * trg_bs * trg_data.train_prop[k])])
          trg_rest_train_idxs = trg_rest_data_idxs[j * trg_rest_bs: (j + 1) * trg_rest_bs]
        elif j == n_batch - 1:
          src_train_idxs.append(src_data_idx[int(j * src_bs * src_data.train_prop[k]):])
          src_rest_train_idxs = src_rest_data_idxs[j * src_rest_bs:]
          trg_train_idxs.append(trg_data_idx[int(j * trg_bs * trg_data.train_prop[k]):])
          trg_rest_train_idxs = trg_rest_data_idxs[j * trg_rest_bs:]

      # source, labeled
      src_batch_train, src_batch_train_lens, \
      src_batch_train_lb, src_batch_train_ohlb = None, None, None, None
      if params.src_le > 0:
        src_batch_train, src_batch_train_lens, \
        src_batch_train_lb, src_batch_train_ohlb = get_batch(params,
                                                             src_train_idxs,
                                                             src_data.train_idxs,
                                                             src_data.train_lens)

      # source, unlabeld
      src_batch_rest_train, src_batch_rest_train_lens, \
      src_batch_rest_train_lb, src_batch_rest_train_ohlb = None, None, None, None
      if params.src_ue > 0:
        src_batch_rest_train, src_batch_rest_train_lens, \
        src_batch_rest_train_lb, src_batch_rest_train_ohlb = get_rest_batch(params,
                                                                            src_rest_train_idxs,
                                                                            src_data.rest_train_idxs,
                                                                            src_data.rest_train_lens,
                                                                            enumerate_discrete)
      # target, labeled
      trg_batch_train, trg_batch_train_lens, \
      trg_batch_train_lb, trg_batch_train_ohlb = None, None, None, None
      if params.trg_le > 0:
        trg_batch_train, trg_batch_train_lens, \
        trg_batch_train_lb, trg_batch_train_ohlb = get_batch(params,
                                                             trg_train_idxs,
                                                             trg_data.train_idxs,
                                                             trg_data.train_lens)

      # target, unlabeled
      trg_batch_rest_train, trg_batch_rest_train_lens, \
      trg_batch_rest_train_lb, trg_batch_rest_train_ohlb = get_rest_batch(params,
                                                                          trg_rest_train_idxs,
                                                                          trg_data.rest_train_idxs,
                                                                          trg_data.rest_train_lens,
                                                                          enumerate_discrete)

      optimizer.zero_grad()
      if params.cldc_train_mode == 'trainenc' and params.adv_training:
        dis_optimizer.zero_grad()
        dis_enc_optimizer.zero_grad()
      m.train()

      if i + 1 <= params.cldc_warm_up_ep:
        m.warm_up = True
      else:
        m.warm_up = False

      src_loss_dict, trg_loss_dict, \
      src_batch_pred, trg_batch_pred = m(src_batch_train, src_batch_train_lens, src_batch_train_lb, src_batch_train_ohlb,
                                         src_batch_rest_train, src_batch_rest_train_lens, src_batch_rest_train_lb, src_batch_rest_train_ohlb,
                                         trg_batch_train, trg_batch_train_lens, trg_batch_train_lb, trg_batch_train_ohlb,
                                         trg_batch_rest_train, trg_batch_rest_train_lens, trg_batch_rest_train_lb, trg_batch_rest_train_ohlb)

      src_batch_acc = .0
      src_batch_acc_cls = [.0, .0, .0, .0]
      if src_batch_train_lb is not None:
        src_batch_acc, src_batch_acc_cls = get_classification_report(params,
                                                                     src_batch_train_lb.data.cpu().numpy(),
                                                                     src_batch_pred.data.cpu().numpy())
      trg_batch_acc = .0
      trg_batch_acc_cls = [.0, .0, .0, .0]
      if trg_batch_train_lb is not None:
        trg_batch_acc, trg_batch_acc_cls = get_classification_report(params,
                                                                     trg_batch_train_lb.data.cpu().numpy(),
                                                                     trg_batch_pred.data.cpu().numpy())
      if src_loss_dict['L_cldc_loss'] < params.cldc_lossth:
        es_flag = True

      out_semicldc(i, j, n_batch,
                   src_loss_dict, trg_loss_dict, src_batch_acc, trg_batch_acc,
                   src_batch_acc_cls, trg_batch_acc_cls,
                   bdev, btest, cdev, ctest, es.num_bad_epochs)

      '''
      #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.grad is not None and p.requires_grad, m.parameters()), 5)
      # debug for gradient
      for p_name, p in m.named_parameters():
        if p.grad is not None and p.requires_grad:
          print(p_name, p.grad.data.norm(2).item())
      '''

      optimizer.step()
      if params.adv_training:
        dis_optimizer.step()
        dis_enc_optimizer.step()
      cur_it += 1

      update_tensorboard(params, writer,
                         src_loss_dict, src_batch_acc, src_batch_acc_cls,
                         trg_loss_dict, trg_batch_acc, trg_batch_acc_cls,
                         cdev, ctest, dev_class_acc, test_class_acc, cur_it)

      if cur_it % params.CLDC_VAL_EVERY == 0:
        sys.stdout.write('\n') 
        sys.stdout.flush()
        # validation
        #cdev, dev_class_acc, dev_cm = test(params, m, trg_data.dev_idxs, trg_data.dev_lens, trg_data.dev_size, trg_data.dev_prop, trg_lang, cm = True)
        # fully zero-shot
        cdev, dev_class_acc, dev_cm = test(params, m, src_data.dev_idxs, src_data.dev_lens, src_data.dev_size, src_data.dev_prop, src_lang, cm = True)
        # copy embeddings from source -> target
        '''
        src_idxs = list(src_data.vocab.inter_vocab_map.keys())
        src_lang_idx = params.lang_dict[src_data.vocab.lang]
        trg_idxs = list(trg_data.vocab.inter_vocab_map.keys())
        trg_lang_idx = params.lang_dict[trg_data.vocab.lang]
        m.xlingva.embeddings.embeddings[trg_lang_idx].embeddings.weight[trg_idxs].data = \
        m.xlingva.embeddings.embeddings[src_lang_idx].embeddings.weight[src_idxs].data
        '''
        # test
        ctest, test_class_acc, test_cm = test(params, m, trg_data.test_idxs, trg_data.test_lens, trg_data.test_size, trg_data.test_prop, trg_lang, cm = True)
        print(dev_cm)
        print(test_cm)
        if es.step(cdev):
          print('\nEarly Stoped.')
          return
        elif es.is_better(cdev, bdev):
          bdev = cdev
          btest = ctest
        # reset bad epochs
        if not es_flag:
          es.num_bad_epochs = 0


def gen_xltask_info(params, m, datas):
  gen_task_info(params, m, datas)
  xlsemicldc_spec_log = ('Adv training: {}\n'.format(params.adv_training) +
                         'src_le = {} src_ue = {} src_alpha = {} '.format(params.src_le, params.src_ue, params.src_cls_alpha) +
                         'trg_le = {} trg_ue = {} trg_alpha = {}\n'.format(params.trg_le, params.trg_ue, params.trg_cls_alpha) +
                         '{}'.format('=' * 80)
                        )
  print(xlsemicldc_spec_log)


def get_optimizer(params, m):
  if params.cldc_train_mode == 'fixenc':
    # do not update encoder
    ps = [p[1] for p in m.named_parameters() if 'xlingva' not in p[0]]
    optimizer = optim.Adam(ps, lr=params.semicldc_init_lr)
    print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
    return optimizer, None, None
  elif params.cldc_train_mode == 'trainenc':
    ps = {p[0]: p[1] for p in m.named_parameters() if 'discriminator' not in p[0] and p[1].requires_grad}
    # freeze parameters
    if params.zs_freeze == 'encoder':
      ps = {p[0]: p[1] for p in ps.items() if
            'embedding' not in p[0]
            and 'xlingva.encoder' not in p[0]
            and 'xlingva.inferer' not in p[0]}
    elif params.zs_freeze == 'embedding':
      ps = {p[0]: p[1] for p in ps.items() if 'embedding' not in p[0]}
    ps = [p[1] for p in ps.items()]
    optimizer = optim.Adam(ps, lr = params.semicldc_init_lr)
    print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
    if params.adv_training:
      dis_ps = [p[1] for p in m.named_parameters() if 'discriminator' in p[0]]
      dis_optimizer = optim.Adam(dis_ps, lr = params.semicldc_init_lr)
      print('Discriminator parameter: {}'.format(sum(p.numel() for p in dis_ps)))
      dis_enc_ps = [p[1] for p in m.named_parameters() if 'encoder' in p[0]]
      dis_enc_optimizer = optim.Adam(dis_enc_ps, lr = params.semicldc_init_lr)
      print('Encoder parameter: {}'.format(sum(p.numel() for p in dis_enc_ps)))
      return optimizer, dis_optimizer, dis_enc_optimizer
    else:
      return optimizer, None, None


def update_tensorboard(params, writer,
                   src_loss_dict, src_batch_acc, src_batch_acc_cls,
                   trg_loss_dict, trg_batch_acc, trg_batch_acc_cls,
                   cdev, ctest, dev_class_acc, test_class_acc, cur_it):
  if writer is None:
    return

  writer.add_scalar("train/src_total_loss", src_loss_dict['src_total_loss'], cur_it)
  writer.add_scalar("train/trg_total_loss", trg_loss_dict['trg_total_loss'], cur_it)
  # classification
  writer.add_scalar("train/src_cldc_loss", src_loss_dict['L_cldc_loss'], cur_it)
  writer.add_scalar("train/src_acc", src_batch_acc, cur_it)
  writer.add_scalar("train/trg_cldc_loss", trg_loss_dict['L_cldc_loss'], cur_it)
  writer.add_scalar("train/trg_acc", trg_batch_acc, cur_it)
  # L
  writer.add_scalar("train/src_LRec", src_loss_dict['L_rec'], cur_it)
  writer.add_scalar("train/src_Lkl", src_loss_dict['L_kld'], cur_it)
  writer.add_scalar("train/src_Lloss", src_loss_dict['L_loss'], cur_it)

  writer.add_scalar("train/trg_LRec", trg_loss_dict['L_rec'], cur_it)
  writer.add_scalar("train/trg_Lkl", trg_loss_dict['L_kld'], cur_it)
  writer.add_scalar("train/trg_Lloss", trg_loss_dict['L_loss'], cur_it)
  # U
  writer.add_scalar("train/src_URec", src_loss_dict['U_rec'], cur_it)
  writer.add_scalar("train/src_Ukl", src_loss_dict['U_kld'], cur_it)
  writer.add_scalar("train/src_Uloss", src_loss_dict['UL_mean_loss'], cur_it)
  writer.add_scalar("train/src_UH", src_loss_dict['H'], cur_it)
  writer.add_scalar("train/src_ukly", src_loss_dict['kldy'], cur_it)

  writer.add_scalar("train/trg_URec", trg_loss_dict['U_rec'], cur_it)
  writer.add_scalar("train/trg_Ukl", trg_loss_dict['U_kld'], cur_it)
  writer.add_scalar("train/trg_Uloss", trg_loss_dict['UL_mean_loss'], cur_it)
  writer.add_scalar("train/trg_UH", trg_loss_dict['H'], cur_it)
  writer.add_scalar("train/trg_ukly", trg_loss_dict['kldy'], cur_it)

  if params.cldc_train_mode == 'trainenc':
    writer.add_scalar("train/src_Lnll", src_loss_dict['L_nll'], cur_it)
    writer.add_scalar("train/src_LHz1", src_loss_dict['L_Hz1'], cur_it)
    writer.add_scalar("train/src_Unll", src_loss_dict['U_nll'], cur_it)
    writer.add_scalar("train/src_UHz1", src_loss_dict['U_Hz1'], cur_it)
    writer.add_scalar("train/src_Lz1kl", src_loss_dict['L_z1kld'], cur_it)
    writer.add_scalar("train/src_Uz1kl", src_loss_dict['U_z1kld'], cur_it)
    writer.add_scalar("train/src_LTKL", src_loss_dict['L_TKL'], cur_it)
    writer.add_scalar("train/src_UTKL", src_loss_dict['U_TKL'], cur_it)

    writer.add_scalar("train/trg_Lnll", trg_loss_dict['L_nll'], cur_it)
    writer.add_scalar("train/trg_LHz1", trg_loss_dict['L_Hz1'], cur_it)
    writer.add_scalar("train/trg_Unll", trg_loss_dict['U_nll'], cur_it)
    writer.add_scalar("train/trg_UHz1", trg_loss_dict['U_Hz1'], cur_it)
    writer.add_scalar("train/trg_Lz1kl", trg_loss_dict['L_z1kld'], cur_it)
    writer.add_scalar("train/trg_Uz1kl", trg_loss_dict['U_z1kld'], cur_it)
    writer.add_scalar("train/trg_LTKL", trg_loss_dict['L_TKL'], cur_it)
    writer.add_scalar("train/trg_UTKL", trg_loss_dict['U_TKL'], cur_it)

  if params.adv_training:
    writer.add_scalar("train/src_Ldis", src_loss_dict['L_dis_loss'], cur_it)
    writer.add_scalar("train/src_Lenc", src_loss_dict['L_enc_loss'], cur_it)
    writer.add_scalar("train/src_Udis", src_loss_dict['U_dis_loss'], cur_it)
    writer.add_scalar("train/src_Uenc", src_loss_dict['U_enc_loss'], cur_it)
    writer.add_scalar("train/trg_Ldis", trg_loss_dict['L_dis_loss'], cur_it)
    writer.add_scalar("train/trg_Lenc", trg_loss_dict['L_enc_loss'], cur_it)
    writer.add_scalar("train/trg_Udis", trg_loss_dict['U_dis_loss'], cur_it)
    writer.add_scalar("train/trg_Uenc", trg_loss_dict['U_enc_loss'], cur_it)

  writer.add_scalar("dev/cur_acc", cdev, cur_it)
  writer.add_scalar("test/cur_acc", ctest, cur_it)
  if dev_class_acc and test_class_acc:
    writer.add_scalars("dev/class_acc", dev_class_acc, cur_it)
    writer.add_scalars("test/class_acc", test_class_acc, cur_it)

  # z1, z2 distance
  writer.add_scalar('train/src_L_l2_dist', src_loss_dict['L_l2_dist'], cur_it)
  writer.add_scalar('train/src_L_cosdist', src_loss_dict['L_cosdist'], cur_it)
  writer.add_scalar('train/src_L_z1z2kld', src_loss_dict['L_z1z2kld'], cur_it)
  writer.add_scalar('train/src_U_l2_dist', src_loss_dict['U_l2_dist'], cur_it)
  writer.add_scalar('train/src_U_cosdist', src_loss_dict['U_cosdist'], cur_it)
  writer.add_scalar('train/src_U_z1z2kld', src_loss_dict['U_z1z2kld'], cur_it)

  writer.add_scalar('train/trg_L_l2_dist', trg_loss_dict['L_l2_dist'], cur_it)
  writer.add_scalar('train/trg_L_cosdist', trg_loss_dict['L_cosdist'], cur_it)
  writer.add_scalar('train/trg_L_z1z2kld', trg_loss_dict['L_z1z2kld'], cur_it)
  writer.add_scalar('train/trg_U_l2_dist', trg_loss_dict['U_l2_dist'], cur_it)
  writer.add_scalar('train/trg_U_cosdist', trg_loss_dict['U_cosdist'], cur_it)
  writer.add_scalar('train/trg_U_z1z2kld', trg_loss_dict['U_z1z2kld'], cur_it)

  # au
  writer.add_scalars('train/src_au',
                     {
                       'src_L_mu1': src_loss_dict['L_mu1'],
                       'src_L_mu2': src_loss_dict['L_mu2'],
                       'src_U_mu1': src_loss_dict['U_mu1'],
                       'src_U_mu2': src_loss_dict['U_mu2'],
                     },
                     cur_it)

  writer.add_scalars('train/trg_au',
                     {
                       'trg_L_mu1': trg_loss_dict['L_mu1'],
                       'trg_L_mu2': trg_loss_dict['L_mu2'],
                       'trg_U_mu1': trg_loss_dict['U_mu1'],
                       'trg_U_mu2': trg_loss_dict['U_mu2'],
                     },
                     cur_it)
