# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
training with monolingual xling corpus
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

from data_model.data_reader import DataReader
from data_model.sw_data_reader import SPMDataReader
from nn_model.xlingva import XlingVA
from utils.early_stopping import EarlyStopping
from train_parallel import get_batch, update_tensorboard, calc_loss_batch
from utils.ios import out_xling
from tensorboardX import SummaryWriter

import pdb


def main(params, vocabs, model_dict):
  # get data 
  datas = get_data(params, vocabs)
  # get model
  m = XlingVA(params, datas, model_dict)
  train(params, m, datas)


def get_data(params, vocabs):
  datas = []
  if params.seg_type == 'word':
    for lang in params.langs:
      datas.append(DataReader(params, vocabs[params.lang_dict[lang]]))
  elif params.seg_type == 'spm':
    # only one MIXED data
    for lang in params.langs:
      datas.append(SPMDataReader(params, vocabs[params.lang_dict[lang]]))
  return datas


def train(params, m, datas):
  es = EarlyStopping(min_delta = params.min_delta, patience = params.patience)

  # optimizer
  ps = [p[1] for p in m.named_parameters() if 'discriminator' not in p[0]]
  print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
  optimizer = optim.Adam(ps, lr = params.init_lr)
  if params.adv_training:
    dis_ps = [p[1] for p in m.named_parameters() if 'discriminator' in p[0]]
    dis_optimizer = optim.Adam(dis_ps, lr = params.init_lr)
    dis_enc_ps = [p[1] for p in m.named_parameters() if 'encoder' in p[0] or 'embedding' in p[0]]
    dis_enc_optimizer = optim.Adam(dis_enc_ps, lr = params.init_lr)

  # all training instances, split between 2 languages, right now the data are balanced
  n_batch = len(datas) * datas[0].train_size // params.bs if len(datas) * datas[0].train_size % params.bs == 0 else len(datas) * datas[0].train_size // params.bs + 1
  data_idxs = {}
  for i, data in enumerate(datas):
    lang = data.vocab.lang
    data_idxs[lang] = list(range(data.train_size))
  
  # number of iterations
  cur_it = 0
  # write to tensorboard
  writer = SummaryWriter('./history/{}'.format(params.log_path)) if params.write_tfboard else None
  
  nll_dev = math.inf
  best_nll_dev = math.inf
  kld_dev = math.inf

  for i in range(params.ep):
    for lang in data_idxs:
      shuffle(data_idxs[lang])
    for j in range(n_batch):
      if params.task == 'xl' or params.task == 'xl-adv':
        lang_idx = j % len(datas)
        data = datas[lang_idx]
        lang = data.vocab.lang
        train_idxs = data_idxs[lang][j // len(datas) * params.bs: (j // len(datas) + 1) * params.bs]
      elif params.task == 'mo':
        lang = params.langs[0]
        lang_idx = params.lang_dict[lang]
        data = datas[lang_idx]
        train_idxs = data_idxs[lang][j * params.bs: (j + 1) * params.bs]
      padded_batch, batch_lens = get_batch(train_idxs, data, data.train_idxs, data.train_lens, params.cuda)

      optimizer.zero_grad()
      if params.adv_training:
        dis_optimizer.zero_grad()
        dis_enc_optimizer.zero_grad()
      m.train()

      nll_batch, kld_batch, ls_dis, ls_enc = m(lang, padded_batch, batch_lens)

      cur_it += 1
      loss_batch, alpha = calc_loss_batch(params, nll_batch, kld_batch, cur_it, n_batch)
      '''
      # add adversarial loss to the encoder
      if cur_it > params.adv_ep * n_batch:
        loss_batch += ls_enc
      '''

      if not params.adv_training:
        loss_batch.backward()
        optimizer.step()
      else:
        ls_dis = ls_dis.mean()
        ls_enc = ls_enc.mean()
        loss_batch = loss_batch + ls_dis + ls_enc
        loss_batch.backward()
        optimizer.step()
        dis_optimizer.step()
        dis_enc_optimizer.step()

      out_xling(i, j, n_batch, loss_batch, nll_batch, kld_batch, best_nll_dev, nll_dev, kld_dev, es.num_bad_epochs, ls_dis = ls_dis, ls_enc = ls_enc)
      update_tensorboard(writer, loss_batch, nll_batch, kld_batch, alpha, nll_dev, kld_dev, cur_it, ls_dis = ls_dis, ls_enc = ls_enc)

      if cur_it % params.VAL_EVERY == 0:
        sys.stdout.write('\n') 
        sys.stdout.flush()
        # validation 
        nll_dev, kld_dev = test(params, m, datas)
        if es.step(nll_dev):
          print('\nEarly Stoped.')
          return
        elif es.is_better(nll_dev, best_nll_dev):
          best_nll_dev = nll_dev
          # save model
          for lang in params.langs:
            lang_idx = params.lang_dict[lang]
            m.save_embedding(params, datas[lang_idx])
          m.save_model(params, datas)


def test(params, m, datas):
  m.eval()
  nll_tot = .0
  kld_tot = .0
  batch_tot = .0
  for lang in params.langs:
    lang_idx = params.lang_dict[lang]
    data = datas[lang_idx]
    n_batch = data.dev_size // params.test_bs if data.dev_size % params.test_bs == 0 else data.dev_size // params.test_bs + 1
    batch_tot += n_batch
    data_idxs = list(range(data.dev_size))

    for k in range(n_batch):
      test_idxs = data_idxs[k * params.test_bs: (k + 1) * params.test_bs]
      # get padded & sorted batch idxs and 
      with torch.no_grad():
        padded_batch, batch_lens = get_batch(test_idxs, data, data.dev_idxs, data.dev_lens, params.cuda)
        nll_batch, kld_batch, _, _ = m(lang, padded_batch, batch_lens)
      nll_tot += nll_batch
      kld_tot += kld_batch

  nll_tot /= batch_tot
  kld_tot /= batch_tot
  return nll_tot.item(), kld_tot.item()
