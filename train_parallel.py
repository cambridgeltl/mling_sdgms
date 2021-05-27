# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
training scripts for parallel input
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

from utils.early_stopping import EarlyStopping
from utils.ios import out_parallel
from tensorboardX import SummaryWriter

import pdb

random.seed(1234)
np.random.seed(1234)


def train(params, m, data_x, data_y):
  es = EarlyStopping(min_delta = params.min_delta, patience = params.patience)

  # optimizer
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr = params.init_learning_rate)
  
  n_batch = data_x.train_size // params.bs if data_x.train_size % params.bs == 0 else data_x.train_size // params.bs + 1
  data_idxs = list(range(data_x.train_size))
  
  # number of iterations
  cur_it = 0
  # write to tensorboard
  writer = SummaryWriter('./history/{}'.format(params.emb_out_path)) if params.write_tfboard else None

  nll_dev = math.inf
  best_nll_dev = math.inf
  kld_dev = math.inf

  for i in range(params.ep):
    shuffle(data_idxs)
    for j in range(n_batch):
      train_idxs = data_idxs[j * params.bs: (j + 1) * params.bs]
      # get padded & sorted batch idxs and 
      padded_batch_x, batch_x_lens = get_batch(train_idxs, data_x, data_x.train_idxs, data_x.train_lens, params.cuda)
      padded_batch_y, batch_y_lens = get_batch(train_idxs, data_y, data_y.train_idxs, data_y.train_lens, params.cuda)

      optimizer.zero_grad()
      m.train()
      nll_batch, kld_batch = m(padded_batch_x, batch_x_lens, padded_batch_y, batch_y_lens)

      cur_it += 1
      loss_batch, alpha = calc_loss_batch(params, nll_batch, kld_batch, cur_it, n_batch)

      loss_batch.backward()
      optimizer.step()

      out_parallel(i, j, n_batch, loss_batch, nll_batch, kld_batch, best_nll_dev, nll_dev, kld_dev, es.num_bad_epochs)
      update_tensorboard(writer, loss_batch, nll_batch, kld_batch, alpha, nll_dev, kld_dev, cur_it)

      if cur_it % params.VAL_EVERY == 0:
        sys.stdout.write('\n') 
        sys.stdout.flush()
        # validation 
        nll_dev, kld_dev = test(params, m, data_x, data_y)
        if es.step(nll_dev):
          print('\nEarly Stoped.')
          return
        elif es.is_better(nll_dev, best_nll_dev):
          best_nll_dev = nll_dev
          # save model
          m.save_embedding(params, data_x, 'x')
          m.save_embedding(params, data_y, 'y')
          m.save_model(params, data_x, data_y, optimizer)


def test(params, m, data_x, data_y):
  m.eval()
  n_batch = data_x.dev_size // params.bs if data_x.dev_size % params.bs == 0 else data_x.dev_size // params.bs + 1
  data_idxs = list(range(data_x.dev_size))

  nll_tot = .0
  kld_tot = .0

  for k in range(n_batch):
    test_idxs = data_idxs[k * params.bs: (k + 1) * params.bs]
    # get padded & sorted batch idxs and 
    with torch.no_grad():
      padded_batch_x, batch_x_lens = get_batch(test_idxs, data_x, data_x.dev_idxs, data_x.dev_lens, params.cuda)
      padded_batch_y, batch_y_lens = get_batch(test_idxs, data_y, data_y.dev_idxs, data_y.dev_lens, params.cuda)

      nll_batch, kld_batch = m(padded_batch_x, batch_x_lens, padded_batch_y, batch_y_lens)
    nll_tot += nll_batch
    kld_tot += kld_batch

  nll_tot /= n_batch
  kld_tot /= n_batch
  return nll_tot, kld_tot


def get_batch(train_idxs, data, text_idxs, text_lens, use_cuda):
  batch_data = [text_idxs[idx] for idx in train_idxs]
  batch_data_lens = text_lens[train_idxs]
  
  # pad idxs
  padded_batch_data = pad_texts(batch_data, data.max_text_len, data.vocab.PAD_ID)

  if use_cuda:
    padded_batch_data = padded_batch_data.cuda()

  return padded_batch_data, batch_data_lens


def pad_texts(text_idxs, max_text_len, PAD_ID):
  padded_text_idxs = []
  for line_idx in text_idxs:
    padded_line_idx = line_idx + [PAD_ID] * (max_text_len - len(line_idx))
    padded_text_idxs.append(padded_line_idx)
  return torch.LongTensor(padded_text_idxs)


def calc_loss_batch(params, nll, kld, cur_it, n_batch):
  if params.vae_type == 'detz':
    return nll
  # Bowman annealing
  # linear
  #alpha = min(1.0, (cur_it - st_it) / ed_it)
  elif params.vae_type == 'sigmoid':
    # sigmoid
    x0 = params.sigmoid_x0ep * n_batch
    alpha = float(1 / (1 + np.exp(-1 * params.sigmoid_k * (cur_it - x0))))
    loss = nll + alpha * kld
  elif params.vae_type == 'beta':
    # linear annealing of alpha
    alpha = cur_it / (n_batch * params.ep)
    loss = nll + params.beta_gamma * torch.abs(kld - alpha * params.beta_C)
  elif params.vae_type == 'standard':
    # vanilla vae
    alpha = 1
    loss = nll + kld
  elif params.vae_type == 'fixa':
    alpha = params.fixed_alpha
    loss = nll + alpha * kld
  elif params.vae_type == 'nokld':
    # only nll
    alpha = 0
    loss = nll
  return loss, alpha


def update_tensorboard(writer, loss, nll, kld, alpha, nll_dev, kld_dev, cur_it, ls_dis = None, ls_enc = None):
  if writer is None:
    return
  writer.add_scalar("NLL", nll.item(), cur_it)
  writer.add_scalar("NLL_DEV", nll_dev, cur_it)
  writer.add_scalar("KLD", kld.item(), cur_it)
  writer.add_scalar("KLD_DEV", kld_dev, cur_it)
  writer.add_scalar("alpha", alpha, cur_it)
  writer.add_scalar("LOSS", loss.item(), cur_it)
  if ls_dis is not None and ls_enc is not None:
    writer.add_scalar("LS_DIS", ls_dis.item(), cur_it)
    writer.add_scalar("LS_ENC", ls_enc.item(), cur_it)


