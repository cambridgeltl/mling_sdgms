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

from nn_model.semicldc_model import SEMICLDCModel
from nn_model.aux_semicldc_model import AUXSEMICLDCModel
from utils.early_stopping import EarlyStopping
from utils import ios
from utils.utils import onehot, enumerate_discrete
from tensorboardX import SummaryWriter
from train_cldc import get_data, get_lang_data, test, gen_task_gen_info, tsne2d, save_model, get_classification_report


import pdb


def main(params, vocabs, model_dict = None, task_model_dict = None, aux = False):
  # get data
  datas = get_data(params, vocabs)
  # get model
  if aux:
    m = AUXSEMICLDCModel(params, datas, model_dict = model_dict, task_model_dict = task_model_dict)
  else:
    m = SEMICLDCModel(params, datas, model_dict = model_dict)
  gen_task_info(params, m, datas)
  train(params, m, datas)


def train(params, m, datas):
  # early stopping
  es = EarlyStopping(mode = 'max', patience = params.cldc_patience)
  # set optimizer
  optimizer = get_optimizer(params, m)
  # get initial parameters
  if params.zs_reg_alpha > 0:
    init_param_dict = {k: v.detach().clone() for k, v in m.named_parameters() if v.requires_grad}

  # training
  train_lang, train_data = get_lang_data(params, datas, training = True)
  # dev & test are in the same lang
  test_lang, test_data = get_lang_data(params, datas)

  n_batch = train_data.train_size // params.cldc_bs if train_data.train_size % params.cldc_bs == 0 else train_data.train_size // params.cldc_bs + 1
  # get the same n_batch for unlabelled data as well
  # batch size for unlabelled data
  rest_cldc_bs = train_data.rest_train_size // n_batch
  # per category
  data_idxs = [list(range(len(train_idx))) for train_idx in train_data.train_idxs]
  rest_data_idxs = list(range(len(train_data.rest_train_idxs)))
 
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
  dev_cm = None
  test_cm = None
  # early stopping warm up flag, start es after train loss below some threshold
  es_flag = False
  # set io function
  out_semicldc = getattr(ios, 'out_semicldc_{}'.format(params.cldc_train_mode))

  for i in range(params.cldc_ep):
    for data_idx in data_idxs:
      shuffle(data_idx)
    shuffle(rest_data_idxs)
    for j in range(n_batch):
      train_idxs = []
      for k, data_idx in enumerate(data_idxs):
        if j < n_batch - 1:
          train_idxs.append(data_idx[int(j * params.cldc_bs * train_data.train_prop[k]): int((j + 1) * params.cldc_bs * train_data.train_prop[k])])
          rest_train_idxs = rest_data_idxs[j * rest_cldc_bs: (j + 1) * rest_cldc_bs]
        elif j == n_batch - 1:
          train_idxs.append(data_idx[int(j * params.cldc_bs * train_data.train_prop[k]):])
          rest_train_idxs = rest_data_idxs[j * rest_cldc_bs:]

      # get batch data
      batch_train, batch_train_lens, batch_train_lb, batch_train_ohlb = get_batch(params, train_idxs, train_data.train_idxs, train_data.train_lens) 
      batch_rest_train, batch_rest_train_lens, batch_rest_train_lb, batch_rest_train_ohlb = get_rest_batch(params, rest_train_idxs, train_data.rest_train_idxs, train_data.rest_train_lens, enumerate_discrete)

      optimizer.zero_grad()
      m.train()

      if i + 1 <= params.cldc_warm_up_ep:
        m.warm_up = True
      else:
        m.warm_up = False

      loss_dict, batch_pred = m(train_lang, 
                                batch_train, batch_train_lens, batch_train_lb, batch_train_ohlb, 
                                batch_rest_train, batch_rest_train_lens, batch_rest_train_lb, batch_rest_train_ohlb)
      # regularization term
      if params.zs_reg_alpha > 0:
        reg_loss = .0
        for k, v in m.named_parameters():
          if k in init_param_dict and v.requires_grad:
            reg_loss += torch.sum((v - init_param_dict[k]) ** 2)
        print(reg_loss.detach())
        reg_loss *= params.zs_reg_alpha / 2
        reg_loss.backward()

      batch_acc, batch_acc_cls = get_classification_report(params, batch_train_lb.data.cpu().numpy(), batch_pred.data.cpu().numpy())

      if loss_dict['L_cldc_loss'] < params.cldc_lossth:
        es_flag = True

      #loss_dict['total_loss'].backward()
      out_semicldc(i, j, n_batch, loss_dict, batch_acc, batch_acc_cls, bdev, btest, cdev, ctest, es.num_bad_epochs)
      
      #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.grad is not None and p.requires_grad, m.parameters()), 5)
      '''
      # debug for gradient
      for p_name, p in m.named_parameters():
        if p.grad is not None and p.requires_grad:
          print(p_name, p.grad.data.norm(2).item())
      '''

      optimizer.step()
      cur_it += 1
      update_tensorboard(params, writer, loss_dict, batch_acc, cdev, ctest, dev_class_acc, test_class_acc, cur_it)
      
      if cur_it % params.CLDC_VAL_EVERY == 0:
        sys.stdout.write('\n') 
        sys.stdout.flush()
        # validation 
        cdev, dev_class_acc, dev_cm = test(params, m, test_data.dev_idxs, test_data.dev_lens, test_data.dev_size, test_data.dev_prop, test_lang, cm = True)
        ctest, test_class_acc, test_cm = test(params, m, test_data.test_idxs, test_data.test_lens, test_data.test_size, test_data.test_prop, test_lang, cm = True)
        print(dev_cm)
        print(test_cm)
        if es.step(cdev):
          print('\nEarly Stoped.')
          # vis
          #if params.cldc_visualize:
            #tsne2d(params, m)
          # vis
          return
        elif es.is_better(cdev, bdev):
          bdev = cdev
          btest = ctest
          #save_model(params, m)
        # reset bad epochs
        if not es_flag:
          es.num_bad_epochs = 0


def get_optimizer(params, m):
  if params.cldc_train_mode == 'fixenc':
    # do not update encoder
    ps = {p[0]: p[1] for p in m.named_parameters() if 'xlingva' not in p[0]}
  elif params.cldc_train_mode == 'trainenc' or params.cldc_train_mode == 'trainae':
    # only add encoder part
    ps = {p[0]: p[1] for p in m.named_parameters() if p[1].requires_grad}
  if params.zs_freeze == 'encoder':
    ps = {p[0]: p[1] for p in ps.items() if 'xlingva.embedding' not in p[0] and 'xlingva.encoder' not in p[0] and 'xlingva.inferer' not in p[0]}
  elif params.zs_freeze == 'embedding':
    ps = {p[0]: p[1] for p in ps.items() if 'embedding' not in p[0]}
  ps = [p[1] for p in ps.items()]
  print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
  optimizer = optim.Adam(ps, lr = params.semicldc_init_lr)
  return optimizer


def get_batch(params, idxs, input_idxs, input_lens):
  if is_list_empty(idxs):
    return None, None, None, None

  batch_x = []
  batch_x_lens = []
  batch_y = []
  batch_yoh = []
  for i, idx in enumerate(idxs):
    # per category
    if idx:
      batch_x.append(input_idxs[i][idx])
      batch_x_lens.append(input_lens[i][idx])
      batch_y.append([i] * len(idx))
      # get onehot y
      batch_yoh.append(onehot(i, params.cldc_label_size).expand(len(idx), -1))
  batch_x = np.concatenate(batch_x)
  batch_x_lens = np.concatenate(batch_x_lens)
  batch_y = np.concatenate(batch_y)
  batch_yoh = np.concatenate(batch_yoh)

  # sort in the descending order
  sorted_len_idxs = np.argsort(-batch_x_lens)
  sorted_batch_x_lens = batch_x_lens[sorted_len_idxs]
  sorted_batch_x = batch_x[sorted_len_idxs]
  sorted_batch_x = torch.LongTensor(sorted_batch_x)
  sorted_batch_y = batch_y[sorted_len_idxs]
  sorted_batch_y = torch.LongTensor(sorted_batch_y)
  sorted_batch_yoh = batch_yoh[sorted_len_idxs]
  sorted_batch_yoh = torch.Tensor(sorted_batch_yoh)

  if params.cuda:
    sorted_batch_x = sorted_batch_x.cuda()
    sorted_batch_y = sorted_batch_y.cuda()
    sorted_batch_yoh = sorted_batch_yoh.cuda()

  return sorted_batch_x, sorted_batch_x_lens, sorted_batch_y, sorted_batch_yoh


def get_rest_batch(params, idxs, input_idxs, input_lens, enumerate_discrete):
  if is_list_empty(idxs):
    return None, None, None, None

  batch_x_lens = input_lens[idxs]
  batch_x = input_idxs[idxs]

  # sort in the descending order
  sorted_len_idxs = np.argsort(-batch_x_lens)
  sorted_batch_x_lens = batch_x_lens[sorted_len_idxs]
  sorted_batch_x = batch_x[sorted_len_idxs]
  sorted_batch_x = torch.LongTensor(sorted_batch_x)
  sorted_batch_yoh = enumerate_discrete(sorted_batch_x, params.cldc_label_size)
  # [0,1,2,3, 0,1,2,3,......]
  sorted_batch_y = sorted_batch_yoh.max(dim = 1)[1]

  if params.cuda:
    sorted_batch_yoh = sorted_batch_yoh.cuda()
    sorted_batch_y = sorted_batch_y.cuda()
    sorted_batch_x = sorted_batch_x.cuda()

  return sorted_batch_x, sorted_batch_x_lens, sorted_batch_y, sorted_batch_yoh


def gen_task_info(params, m, datas):
  gen_task_gen_info(params, m, datas)
  semicldc_spec_log = ('Init lr: {}\n'.format(params.semicldc_init_lr) + 
                       'X Y condition: {}\n'.format(params.semicldc_cond_type) + 
                       'Y prior: {}\n'.format(params.semicldc_yprior_type) + 
                       'U batch size: {}\n'.format(params.semicldc_U_bs) + 
                       '{}'.format('=' * 80)
                      )
  print(semicldc_spec_log)


def is_list_empty(in_list):
  if isinstance(in_list, list):  # Is a list
    return all(map(is_list_empty, in_list))
  return False


def update_tensorboard(params, writer, loss_dict, cldc_acc_batch, cdev, ctest, dev_class_acc, test_class_acc, cur_it):
  if writer is None:
    return
    
  writer.add_scalar("train/total_loss", loss_dict['total_loss'], cur_it)
  # classification
  writer.add_scalar("train/cldc_loss", loss_dict['L_cldc_loss'], cur_it)
  writer.add_scalar("train/acc", cldc_acc_batch, cur_it)
  # L
  writer.add_scalar("train/LRec", loss_dict['L_rec'], cur_it)
  writer.add_scalar("train/Lkl", loss_dict['L_kld'], cur_it)
  writer.add_scalar("train/Lloss", loss_dict['L_loss'], cur_it)
  # U
  writer.add_scalar("train/URec", loss_dict['U_rec'], cur_it)
  writer.add_scalar("train/Ukl", loss_dict['U_kld'], cur_it)
  writer.add_scalar("train/Uloss", loss_dict['UL_mean_loss'], cur_it)
  writer.add_scalar("train/UH", loss_dict['H'], cur_it)
  writer.add_scalar("train/ukly", loss_dict['kldy'], cur_it)

  if params.cldc_train_mode == 'trainenc':
    writer.add_scalar("train/Lnll", loss_dict['L_nll'], cur_it)
    writer.add_scalar("train/LHz1", loss_dict['L_Hz1'], cur_it)
    writer.add_scalar("train/Unll", loss_dict['U_nll'], cur_it)
    writer.add_scalar("train/UHz1", loss_dict['U_Hz1'], cur_it)
    writer.add_scalar("train/Lz1kl", loss_dict['L_z1kld'], cur_it)
    writer.add_scalar("train/Uz1kl", loss_dict['U_z1kld'], cur_it)
    writer.add_scalar("train/LTKL", loss_dict['L_TKL'], cur_it)
    writer.add_scalar("train/UTKL", loss_dict['U_TKL'], cur_it)

  writer.add_scalar("dev/cur_acc", cdev, cur_it)
  writer.add_scalar("test/cur_acc", ctest, cur_it)
  if dev_class_acc and test_class_acc:
    writer.add_scalars("dev/class_acc", dev_class_acc, cur_it)
    writer.add_scalars("test/class_acc", test_class_acc, cur_it)

  # z1, z2 distance
  writer.add_scalar('train/L_l2_dist', loss_dict['L_l2_dist'], cur_it)
  writer.add_scalar('train/L_cosdist', loss_dict['L_cosdist'], cur_it)
  writer.add_scalar('train/L_z1z2kld', loss_dict['L_z1z2kld'], cur_it)
  writer.add_scalar('train/U_l2_dist', loss_dict['U_l2_dist'], cur_it)
  writer.add_scalar('train/U_cosdist', loss_dict['U_cosdist'], cur_it)
  writer.add_scalar('train/U_z1z2kld', loss_dict['U_z1z2kld'], cur_it)

  # au
  writer.add_scalars('train/au',
                     {
                       'L_mu1': loss_dict['L_mu1'],
                       'L_mu2': loss_dict['L_mu2'],
                       'U_mu1': loss_dict['U_mu1'],
                       'U_mu2': loss_dict['U_mu2'],
                     },
                     cur_it)

