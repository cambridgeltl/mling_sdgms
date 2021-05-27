# -*- coding: UTF-8 -*-
# !/usr/bin/python3
"""
Model for SEMI-CLDC task
"""

# ************************************************************
# Imported Libraries
# ************************************************************
import math
import numpy as np
import sympy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_model.xlingva import XlingVA
from nn_model.inferer import Inferer
from nn_model.mlp.cldc_classifier import CLDCClassifier
from utils.logpdfs import multi_diag_normal, cal_kl_gau1, cal_kl_gau2, cal_kl_gau1_fb, \
  cal_kl_gau2_fb

import pdb

const = np.float128((sympy.log(2 * sympy.pi)).evalf(64))

# fb
l_z1_fb = 1.0
l_z2_fb = 1.0
u_z1_fb = 0.1
u_z2_fb = 0.1


class SEMICLDCModel(nn.Module):
  def __init__(self, params, data_list, model_dict=None):
    super(SEMICLDCModel, self).__init__()
    # label size 
    self.label_size = params.cldc_label_size
    # batch size for U / label size 
    self.bs_u = params.semicldc_U_bs // self.label_size
    # alpha for cldc classifier
    self.semicldc_classifier_alpha = params.semicldc_classifier_alpha

    # get y prior
    self.yprior = self.get_yprior(params, data_list)
    # xling vae
    self.xlingva = XlingVA(params, data_list, model_dict=model_dict)

    self.init_semicldc_cond = getattr(self, 'init_semicldc_cond_{}'.format(params.semicldc_cond_type))
    # initialize the setting of how to combine x & y and z & y
    self.init_semicldc_cond(params)

    # cldc MLP
    self.cldc_classifier = CLDCClassifier(params, params.cldc_classifier_config)

    # functions
    self.forward = getattr(self, 'forward_{}'.format(params.cldc_train_mode))
    self.get_z1 = getattr(self, 'get_z1_{}'.format(params.cldc_train_mode))
    # x & y
    self.get_z1y = getattr(self, 'get_z1y_{}'.format(params.semicldc_cond_type))
    # z & y
    self.get_z2y = getattr(self, 'get_z2y_{}'.format(params.semicldc_cond_type))
    # calculate kl of z2
    self.kl_z2 = getattr(self, 'kl_z2_{}'.format(params.semicldc_cond_type))

    self.step = 0
    self.anneal_warm_up = params.semicldc_anneal_warm_up

    # warm up stage
    self.warm_up = False

    self.use_cuda = params.cuda
    if self.use_cuda:
      self.cuda()


  def init_model(self, model_dict):
    if model_dict is None:
      return
    else:
      # 3. load the new state dict
      # parameter names need to be exactly the same
      self.load_state_dict(model_dict)

  def train_classifier(self, lang, batch_in, batch_lens, batch_lb, training=True):
    # x -> hid_x -> mu1, logva1 -> z1
    mu1, logvar1, z1, x, loss_dis, loss_enc = self.get_z1(lang, batch_in, batch_lens)

    # cldc loss for training
    cldc_loss, pred_p, pred = self.cldc_classifier(z1, y=batch_lb, training=training)

    if training:
      L_dict = defaultdict(float)
      if not hasattr(self, 'adv_training') or self.adv_training is False:
        cldc_loss.mean().backward()
        L_dict['L_dis_loss'] = loss_dis
        L_dict['L_enc_loss'] = loss_enc
      elif self.adv_training is True:
        (cldc_loss.mean() +
         loss_dis.mean() +
         loss_enc.mean()).backward()
        L_dict['L_dis_loss'] = loss_dis.mean().item()
        L_dict['L_enc_loss'] = loss_enc.mean().item()
      L_dict['L_cldc_loss'] = cldc_loss.mean().item()
      return L_dict, pred
    else:
      return cldc_loss, pred_p, pred

  def get_yprior(self, params, data_list):
    if params.semicldc_yprior_type == 'uniform':
      # prior scores of y
      yprior_score = torch.ones(self.label_size)
      # uniform distribution
      m = nn.LogSoftmax(dim=-1)
      yprior = m(yprior_score)
    elif params.semicldc_yprior_type == 'train_prop':
      # same distribution as lablled training data
      train_prop = data_list[params.lang_dict[params.cldc_langs[0]]].train_prop
      yprior = torch.log(torch.tensor(train_prop + 1e-32, dtype=torch.float, requires_grad=False))

    if params.cuda:
      yprior = yprior.cuda()

    return yprior

  def init_semicldc_cond_concat(self, params):
    # concat directly z1 and one hot y
    self.z1y_z2 = Inferer(params, in_dim=params.z_dim + self.label_size)
    self.z2y_z1 = Inferer(params, in_dim=params.z_dim + self.label_size)

  def init_semicldc_cond_transconcat(self, params):
    # trans one hot y to dense y, then concat z1 and y
    self.yohtoy = nn.Linear(self.label_size, params.z_dim)
    self.z1y_z2 = Inferer(params, in_dim=params.z_dim + params.z_dim)
    self.z2y_z1 = Inferer(params, in_dim=params.z_dim + params.z_dim)

  def init_semicldc_cond_transadd(self, params):
    # trans one_hot_y to dense_y, then add dense_y to z1
    self.leakyrelu = nn.LeakyReLU()
    self.yohtoy_toz2 = nn.Linear(self.label_size, params.z_dim)
    self.hbn_z1y = nn.BatchNorm1d(params.z_dim)
    self.yohtoy_toz1 = nn.Linear(self.label_size, params.z_dim)
    self.hbn_z2y = nn.BatchNorm1d(params.z_dim)
    self.z1y_z2 = Inferer(params, in_dim=params.z_dim)
    self.z2y_z1 = Inferer(params, in_dim=params.z_dim)

  def init_semicldc_cond_gmix(self, params):
    # concat
    # concat z1 and y
    self.z1y_z2 = Inferer(params, in_dim=params.z_dim + self.label_size)
    # p(z2 | y)
    self.y_z2 = Inferer(params, in_dim=self.label_size)
    # p(z1 | z2)
    self.z2y_z1 = Inferer(params, in_dim=params.z_dim)
    '''
    # transadd
    # trans one_hot_y to dense_y, then add dense_y to z1
    self.yohtoy = nn.Linear(self.label_size, params.z_dim)
    self.z1y_z2 = Inferer(params, in_dim = params.z_dim)
    self.y2z2 = Inferer(params, in_dim = params.z_dim)
    self.z2y_z1 = Inferer(params, in_dim = params.z_dim)
    '''


  def init_semicldc_cond_gmix_transadd(self, params):
    # trans one_hot_y to dense_y, then add dense_y to z1
    self.leakyrelu = nn.LeakyReLU()
    self.yohtoy_toz2 = nn.Linear(self.label_size, params.z_dim)
    self.hbn_z1y = nn.BatchNorm1d(params.z_dim)
    self.z1y_z2 = Inferer(params, in_dim=params.z_dim)

    self.y_z2 = Inferer(params, in_dim=params.cldc_label_size)
    self.z2y_z1 = Inferer(params, in_dim=params.z_dim)


  def forward_trainenc(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb, batch_uin,
                       batch_ulens, batch_ulb, batch_uohlb):
    # warm up
    if self.warm_up:
      return self.train_classifier(lang, batch_in, batch_lens, batch_lb)

    # au
    self.L_mu1, self.U_mu1 = [], []
    self.L_mu2, self.U_mu2 = [], []
    # z1, z2 distance
    self.L_l2dist, self.L_cosdist, self.L_z1z2kld = .0, .0, .0
    self.U_l2dist, self.U_cosdist, self.U_z1z2kld = .0, .0, .0

    # calculate L loss and classfication loss
    L_dict, L_pred = self.forward_L_trainenc(lang, batch_in, batch_lens, batch_lb, batch_ohlb,
                                             cls_alpha=self.semicldc_classifier_alpha)

    # calculate U loss 
    U_dict = self.forward_U_trainenc(lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb)

    # merge two dicts
    loss_dict = {**L_dict, **U_dict}
    self.step += 1

    # z1, z2 distance
    loss_dict['L_l2_dist'], loss_dict['L_cosdist'], loss_dict[
      'L_z1z2kld'] = self.L_l2dist, self.L_cosdist, self.L_z1z2kld
    loss_dict['U_l2_dist'], loss_dict['U_cosdist'], loss_dict[
      'U_z1z2kld'] = self.U_l2dist, self.U_cosdist, self.U_z1z2kld

    # total MEAN loss 
    loss_dict['total_loss'] = loss_dict['L_loss_trainenc'] + loss_dict[
      'U_loss_trainenc'] + self.semicldc_classifier_alpha * loss_dict['L_cldc_loss']

    # au
    loss_dict['L_mu1'] = calc_au(self.L_mu1)[0]
    loss_dict['L_mu2'] = calc_au(self.L_mu2)[0]
    loss_dict['U_mu1'] = calc_au(self.U_mu1)[0]
    loss_dict['U_mu2'] = calc_au(self.U_mu2)[0]
    # au
    print()
    print('L_mu1: {}'.format(loss_dict['L_mu1']))
    print('L_mu2: {}'.format(loss_dict['L_mu2']))
    print('U_mu1: {}'.format(loss_dict['U_mu1']))
    print('U_mu2: {}'.format(loss_dict['U_mu2']))

    return loss_dict, L_pred


  def forward_L_trainenc(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb, le=1.0,
                         cls_alpha=0.1):
    L_dict, L_pred = self.forward_L_trainenc_batch(lang, batch_in, batch_lens, batch_lb, batch_ohlb)

    # calculate all necessary losses
    L_dict = self.cal_L_trainenc(L_dict)

    # backward
    L_dict = self.backward_L_trainenc(L_dict, le, cls_alpha)

    return L_dict, L_pred


  def forward_L_trainenc_batch(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb):
    L_dict, L_pred, mu1, logvar1, z1, rec_mu1, rec_logvar1 = self.forward_L_fixenc_batch(lang,
                                                                                         batch_in,
                                                                                         batch_lens,
                                                                                         batch_lb,
                                                                                         batch_ohlb)

    # nll_loss
    L_dict['L_nll'] = self.xlingva.decoder(lang, z1, batch_in, reduction=None)
    # H(q(z1|x))
    # k/2 + k/2 log(2pi) + 1/2 log(|covariance|)
    # L_dict['L_Hz1'] = -multi_diag_normal(z1, mu1, logvar1)
    L_dict['L_Hz1'] = mu1.shape[1] / 2.0 * (1 + const) + 1 / 2.0 * logvar1.sum(dim=-1)
    # regroup
    L_dict['L_z1kld'] = cal_kl_gau2(mu1, logvar1, rec_mu1, rec_logvar1)
    # fb
    L_dict['L_z1kld_fb'] = cal_kl_gau2_fb(mu1, logvar1, rec_mu1, rec_logvar1, l_z1_fb)

    return L_dict, L_pred

  def cal_L_trainenc(self, L_dict):
    lkld_fix = 5.0
    lz1kld_fix = 5.0

    L_dict = self.cal_L_fixenc(L_dict)
    # L_dict['L_loss_trainenc'] = L_dict['L_loss'] + lnll * L_dict['L_nll'] - lz1kl * L_dict['L_Hz1']
    # regroup
    '''
    kl_weight_z1 = get_cyclic_weight(self.step, self.cyclic_period)
    print()
    print(kl_weight_z1)
    kl_weight_z2 = get_cyclic_weight(self.step, self.cyclic_period)
    '''
    '''
    kl_weight_z1 = min(1.0, self.step / self.anneal_warm_up)
    kl_weight_z2 = min(1.0, self.step / self.anneal_warm_up)
    '''
    kl_weight_z1 = 1.0
    kl_weight_z2 = 1.0
    L_dict['L_loss_trainenc'] = L_dict['L_nll'] + kl_weight_z2 * L_dict['L_kld'] + kl_weight_z1 * \
                                L_dict['L_z1kld'] - L_dict['L_yprior']
    # L_dict['L_loss_trainenc'] = L_dict['L_nll'] + torch.abs(lkld_fix - L_dict['L_kld']) + torch.abs(lz1kld_fix - L_dict['L_z1kld']) - L_dict['L_yprior']
    L_dict['L_TKL'] = L_dict['L_kld'] + L_dict['L_z1kld'] - L_dict['L_yprior']

    # fb
    '''
    kl_weight_nll = min(1.0, self.step / self.anneal_warm_up)
    kl_weight_z1 = get_cyclic_weight(self.step, self.cyclic_period)
    print()
    print(kl_weight_z1)
    kl_weight_z2 = get_cyclic_weight(self.step, self.cyclic_period)
    kl_weight_nll = 1.0
    kl_weight_z1 = 1.0
    kl_weight_z2 = 1.0
    L_dict['L_loss_trainenc'] = kl_weight_nll * L_dict['L_nll'] + kl_weight_z2 * L_dict['L_kld_fb'] + kl_weight_z1 * L_dict['L_z1kld_fb'] - L_dict['L_yprior']
    '''

    # autoencoding wo KL
    # L_dict['L_loss_trainenc'] = L_dict['L_nll']

    return L_dict

  def backward_L_trainenc(self, L_dict, e, alpha):
    '''
    # annealing
    total_step = 5000 * 2
    alpha = self.semicldc_classifier_alpha - (self.semicldc_classifier_alpha - 0.1) * (
            self.step / total_step)
    print()
    print(alpha)
    # cyclic annealing
    # number of steps for increasing
    total_step = 100 * 2
    cur_step = self.step % total_step
    alpha = self.semicldc_classifier_alpha - (self.semicldc_classifier_alpha - 0.1) * (
            cur_step / total_step)
    print()
    print(alpha)
    '''

    if not hasattr(self, 'adv_training') or self.adv_training is False:
      (e * (L_dict['L_loss_trainenc'].mean())
       + alpha * L_dict['L_cldc_loss'].mean()).backward()
    elif self.adv_training is True:
      (e * (L_dict['L_loss_trainenc'].mean()) +
       alpha * L_dict['L_cldc_loss'].mean() +
       L_dict['L_dis_loss'].mean() +
       L_dict['L_enc_loss'].mean()
       ).backward()

    # autoencoding wo KL
    # (L_dict['L_loss_trainenc'].mean()).backward()

    # get mean().item(), reduce memory
    L_dict = {k: (v.mean().item() if torch.is_tensor(v) else float(v))
              for k, v in L_dict.items()}

    return L_dict

  def forward_U_trainenc(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb, ue=1.0):
    U_dict = defaultdict(float)

    cur_bs = batch_uin.shape[0]
    n_bs = math.ceil(cur_bs / self.bs_u)

    for i in range(n_bs):
      U_dict_batch, U_pred_p = self.forward_U_trainenc_batch(lang,
                                                             batch_uin[
                                                             i * self.bs_u: (i + 1) * self.bs_u],
                                                             batch_ulens[
                                                             i * self.bs_u: (i + 1) * self.bs_u],
                                                             batch_ulb[
                                                             i * self.bs_u * self.label_size:
                                                             (i + 1) * self.bs_u * self.label_size],
                                                             batch_uohlb[
                                                             i * self.bs_u * self.label_size:
                                                             (i + 1) * self.bs_u * self.label_size])
      # calculate all necessary losses
      U_dict_batch = self.cal_U_trainenc(U_dict_batch, U_pred_p)
      # backward
      U_dict_batch = self.backward_U_trainenc(U_dict_batch, cur_bs, ue)
      U_dict = {k: (U_dict[k] + v) for k, v in U_dict_batch.items()}

      # z1, z2 distance
    self.U_l2dist /= n_bs
    self.U_cosdist /= n_bs
    self.U_z1z2kld /= n_bs

    U_dict = {k: v / cur_bs for k, v in U_dict.items()}

    return U_dict

  def forward_U_trainenc_batch(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb):
    U_dict, U_pred_p, mu1, logvar1, z1, dup_mu1, dup_logvar1, rec_mu1, rec_logvar1 = self.forward_U_fixenc_batch(
      lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb)

    U_dict['U_nll'] = self.xlingva.decoder(lang, z1, batch_uin, reduction=None)
    # H(q(z1|x))
    # k/2 + k/2 log(2pi) + 1/2 log(|covariance|)
    # U_dict['U_Hz1'] = -multi_diag_normal(z1, mu1, logvar1)
    U_dict['U_Hz1'] = mu1.shape[1] / 2.0 * (1 + const) + 1 / 2.0 * logvar1.sum(dim=-1)
    # regroup
    U_dict['U_z1kld'] = cal_kl_gau2(dup_mu1, dup_logvar1, rec_mu1, rec_logvar1)
    # fb
    U_dict['U_z1kld_fb'] = cal_kl_gau2_fb(dup_mu1, dup_logvar1, rec_mu1, rec_logvar1, u_z1_fb)

    return U_dict, U_pred_p

  def cal_U_trainenc(self, U_dict, U_pred_p):
    ukld_fix = 5.0
    uz1kld_fix = 5.0

    U_dict = self.cal_U_fixenc(U_dict, U_pred_p)

    # U_dict['U_loss_trainenc'] = U_dict['U_loss'] + unll * U_dict['U_nll'] - uz1kl * U_dict['U_Hz1']
    # regroup
    U_dict['U_z1kld'] = torch.sum((U_dict['U_z1kld'] * U_pred_p).view(-1, self.label_size), dim=1)
    '''
    kl_weight_z1 = get_cyclic_weight(self.step, self.cyclic_period)
    kl_weight_z2 = get_cyclic_weight(self.step, self.cyclic_period)
    '''
    '''
    kl_weight_z1 = min(1.0, self.step / self.anneal_warm_up)
    kl_weight_z2 = min(1.0, self.step / self.anneal_warm_up)
    '''
    kl_weight_z1 = 1.0
    kl_weight_z2 = 1.0
    U_dict['U_loss_trainenc'] = U_dict['U_nll'] + kl_weight_z2 * U_dict['U_kld'] + kl_weight_z1 * \
                                U_dict['U_z1kld'] + U_dict['kldy']
    # U_dict['U_loss_trainenc'] = U_dict['U_nll'] + torch.abs(ukld_fix - U_dict['U_kld']) + torch.abs(uz1kld_fix - U_dict['U_z1kld']) + U_dict['kldy']
    U_dict['U_TKL'] = U_dict['U_kld'] + U_dict['U_z1kld'] + U_dict['kldy']

    '''
    # fb
    U_dict['U_z1kld_fb'] = torch.sum((U_dict['U_z1kld_fb'] * U_pred_p).view(-1, self.label_size), dim = 1)
    #kl_weight_nll = min(1.0, self.step / self.anneal_warm_up)
    #kl_weight_z1 = get_cyclic_weight(self.step, self.cyclic_period)
    #kl_weight_z2 = get_cyclic_weight(self.step, self.cyclic_period)
    kl_weight_nll = 1.0
    kl_weight_z1 = 1.0
    kl_weight_z2 = 1.0
    kl_weight_y = 1.0
    U_dict['U_loss_trainenc'] = kl_weight_nll * U_dict['U_nll'] + kl_weight_z2 * U_dict['U_kld_fb'] + kl_weight_z1 * U_dict['U_z1kld_fb'] + kl_weight_y * U_dict['kldy']
    '''

    # autoencoding wo KL
    # U_dict['U_loss_trainenc'] = U_dict['U_nll']

    return U_dict

  def backward_U_trainenc(self, U_dict, cur_bs, e):
    # backward
    if not hasattr(self, 'adv_training') or self.adv_training is False:
      (e * U_dict['U_loss_trainenc'].sum() / cur_bs).backward()
    elif self.adv_training is True:
      (e * U_dict['U_loss_trainenc'].sum() / cur_bs +
       U_dict['U_dis_loss'].sum() / cur_bs +
       U_dict['U_enc_loss'].sum() / cur_bs).backward()

    # get sum().item()
    U_dict = {k: (v.sum().item() if torch.is_tensor(v) else float(v))
              for k, v in U_dict.items()}

    return U_dict


  def forward_fixenc(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb, batch_uin, batch_ulens,
                     batch_ulb, batch_uohlb):
    # au
    self.L_mu1, self.U_mu1 = [], []
    self.L_mu2, self.U_mu2 = [], []
    # z1, z2 distance
    self.L_l2dist, self.L_cosdist, self.L_z1z2kld = .0, .0, .0
    self.U_l2dist, self.U_cosdist, self.U_z1z2kld = .0, .0, .0

    # calculate L loss and classfication loss
    L_dict, L_pred = self.forward_L_fixenc(lang, batch_in, batch_lens, batch_lb, batch_ohlb)

    # calculate U loss 
    U_dict = self.forward_U_fixenc(lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb)

    # merge two dicts
    loss_dict = {**L_dict, **U_dict}
    self.step += 1

    # z1, z2 distance
    loss_dict['L_l2_dist'], loss_dict['L_cosdist'], loss_dict[
      'L_z1z2kld'] = self.L_l2dist, self.L_cosdist, self.L_z1z2kld
    loss_dict['U_l2_dist'], loss_dict['U_cosdist'], loss_dict[
      'U_z1z2kld'] = self.U_l2dist, self.U_cosdist, self.U_z1z2kld

    # total MEAN loss 
    # total_loss = L_loss.mean()
    # total_loss = 0.1 * L_cldc_loss.mean()
    # total_loss = L_loss.mean() + 0.1 * L_cldc_loss.mean()
    # total_loss = L_loss + U_loss
    loss_dict['total_loss'] = loss_dict['L_loss'] + loss_dict[
      'U_loss'] + self.semicldc_classifier_alpha * loss_dict['L_cldc_loss']

    # au
    loss_dict['L_mu1'] = calc_au(self.L_mu1)[0]
    loss_dict['L_mu2'] = calc_au(self.L_mu2)[0]
    loss_dict['U_mu1'] = calc_au(self.U_mu1)[0]
    loss_dict['U_mu2'] = calc_au(self.U_mu2)[0]

    return loss_dict, L_pred


  def forward_L_fixenc(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb):
    L_dict, L_pred, _, _, _, _, _ = self.forward_L_fixenc_batch(lang, batch_in, batch_lens,
                                                                batch_lb, batch_ohlb)

    # calculate all necessary losses
    L_dict = self.cal_L_fixenc(L_dict)
    # backward
    L_dict = self.backward_L_fixenc(L_dict)

    return L_dict, L_pred


  def forward_L_fixenc_batch(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb):
    # x -> hid_x -> mu1, logva1 -> z1
    mu1, logvar1, z1, hid, loss_dis, loss_enc = self.get_z1(lang, batch_in, batch_lens)
    # cldc loss for training
    cldc_loss, _, pred = self.cldc_classifier(z1, y=batch_lb, training=True)
    # z1y -> z2
    mu2, logvar2, z2 = self.get_z2(z1, batch_ohlb)

    # au
    self.L_mu1.append(mu1)
    self.L_mu2.append(mu2)

    # z1, z2 distance
    self.L_z1z2kld += 0.5 * torch.mean(
      torch.sum(logvar1 - logvar2 - 1 + ((mu2 - mu1).pow(2) + logvar2.exp()) / logvar1.exp(),
                dim=1)).item()
    self.L_l2dist += torch.mean(torch.sqrt(torch.sum(((z1 - z2) ** 2), dim=1))).item()
    self.L_cosdist += torch.mean(F.cosine_similarity(z1, z2)).item()

    # reconstruct z1 from z2 
    rec_loss, rec_mu1, rec_logvar1 = self.z2_rec_z1(z1, z2, batch_ohlb)
    # kl divergence of z2
    kld = self.kl_z2(mu2, logvar2, batch_ohlb)
    # get y prior
    yprior = batch_ohlb @ self.yprior

    # fb
    kld_fb = cal_kl_gau1_fb(mu2, logvar2, l_z2_fb)

    # fb
    L_dict = {
      'L_rec': rec_loss,
      'L_kld': kld,
      'L_yprior': yprior,
      'L_cldc_loss': cldc_loss,
      'L_kld_fb': kld_fb,
      'L_dis_loss': loss_dis,
      'L_enc_loss': loss_enc,
    }
    return L_dict, pred, mu1, logvar1, z1, rec_mu1, rec_logvar1


  def kl_z2_gmix(self, mu_post, logvar_post, batch_ohlb):
    # concat
    mu_prior, logvar_prior = self.y2z2(batch_ohlb)
    kld = 0.5 * torch.sum(logvar_prior - logvar_post - 1 + (
              (mu_post - mu_prior).pow(2) + logvar_post.exp()) / logvar_prior.exp(), dim=1)
    '''
    # transadd
    y = self.yohtoy(batch_ohlb)
    mu_prior, logvar_prior = self.y2z2(y)
    kld = 0.5 * torch.sum(logvar_prior - logvar_post - 1 + ((mu_post - mu_prior).pow(2) + logvar_post.exp()) / logvar_prior.exp(), dim = 1)
    '''
    return kld

  def kl_z2_gmix_transadd(self, mu_post, logvar_post, batch_ohlb):
    # transadd
    mu_prior, logvar_prior = self.y_z2(batch_ohlb)
    kld = 0.5 * torch.sum(logvar_prior - logvar_post - 1 + (
              (mu_post - mu_prior).pow(2) + logvar_post.exp()) / logvar_prior.exp(), dim=1)
    return kld

  def kl_z2_general(self, mu_post, logvar_post, batch_ohlb):
    kld = -0.5 * torch.sum(1 + logvar_post - mu_post.pow(2) - logvar_post.exp(), dim=1)
    return kld

  def kl_z2_transadd(self, mu_post, logvar_post, batch_ohlb):
    return self.kl_z2_general(mu_post, logvar_post, batch_ohlb)

  def cal_L_fixenc(self, L_dict):
    # L
    L_dict['L_loss'] = L_dict['L_rec'] + L_dict['L_kld'] - L_dict['L_yprior']
    # L_dict['L_loss'] = lrec * L_dict['L_rec'] + lkld * torch.abs(lkld_fix - L_dict['L_kld']) - L_dict['L_yprior']
    # L_loss = L_rec + min(1.0, self.step / 1000) * L_kld - L_yprior

    return L_dict

  def backward_L_fixenc(self, L_dict):
    # backprop
    (L_dict['L_loss'].mean() + self.semicldc_classifier_alpha * L_dict[
      'L_cldc_loss'].mean()).backward()

    # get mean().item(), reduce memory
    L_dict = {k: v.mean().item() for k, v in L_dict.items()}

    return L_dict


  def forward_U_fixenc(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb):
    U_dict = defaultdict(float)

    cur_bs = batch_uin.shape[0]
    n_bs = math.ceil(cur_bs / self.bs_u)

    for i in range(n_bs):
      U_dict_batch, U_pred_p, _, _, _, _, _, _, _ = self.forward_U_fixenc_batch(lang,
                                                                                batch_uin[
                                                                                i * self.bs_u: (
                                                                                                         i + 1) * self.bs_u],
                                                                                batch_ulens[
                                                                                i * self.bs_u: (
                                                                                                         i + 1) * self.bs_u],
                                                                                batch_ulb[
                                                                                i * self.bs_u * self.label_size: (
                                                                                                                           i + 1) * self.bs_u * self.label_size],
                                                                                batch_uohlb[
                                                                                i * self.bs_u * self.label_size: (
                                                                                                                           i + 1) * self.bs_u * self.label_size])

      # calculate all necessary losses
      U_dict_batch = self.cal_U_fixenc(U_dict_batch, U_pred_p)
      # backward
      U_dict_batch = self.backward_U_fixenc(U_dict_batch, cur_bs)
      U_dict = {k: (U_dict[k] + v) for k, v in U_dict_batch.items()}

    U_dict = {k: v / cur_bs for k, v in U_dict.items()}

    return U_dict

  def forward_U_fixenc_batch(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb):
    mu1, logvar1, z1, hid, loss_dis, loss_enc = self.get_z1(lang, batch_uin, batch_ulens)
    # use classifier to infer
    _, pred_p, _ = self.cldc_classifier(z1, y=None, training=True)
    # gumbel softmax
    # duplicate z1, enumerate y
    # bs * label_size, z_dim
    dup_z1 = self.enumerate_label(z1, batch_uohlb)
    dup_mu1 = self.enumerate_label(mu1, batch_uohlb)
    dup_logvar1 = self.enumerate_label(logvar1, batch_uohlb)
    # z1y -> z2
    mu2, logvar2, z2 = self.get_z2(dup_z1, batch_uohlb)

    self.U_mu1.append(dup_mu1)
    self.U_mu2.append(mu2)

    # z1, z2 distance
    self.U_z1z2kld += 0.5 * torch.mean(torch.sum(
      dup_logvar1 - logvar2 - 1 + ((mu2 - dup_mu1).pow(2) + logvar2.exp()) / dup_logvar1.exp(),
      dim=1)).item()
    self.U_l2dist += torch.mean(torch.sqrt(torch.sum(((dup_z1 - z2) ** 2), dim=1))).item()
    self.U_cosdist += torch.mean(F.cosine_similarity(dup_z1, z2)).item()

    # reconstruct z1 from z2 
    rec_loss, rec_mu1, rec_logvar1 = self.z2_rec_z1(dup_z1, z2, batch_uohlb)
    # kl divergence of z2
    kld = self.kl_z2(mu2, logvar2, batch_uohlb)
    # get y prior
    yprior = batch_uohlb @ self.yprior

    # calculate H(q(y | x ))
    H = -torch.sum(torch.mul(pred_p, torch.log(pred_p + 1e-32)), dim=1)

    # bs * label_size 
    pred_p = pred_p.view(-1)

    # fb
    kld_fb = cal_kl_gau1_fb(mu2, logvar2, u_z2_fb)

    # fb
    U_dict = {
      'U_rec': rec_loss,
      'U_kld': kld,
      'U_yprior': yprior,
      'H': H,
      'U_kld_fb': kld_fb,
      'U_dis_loss': loss_dis,
      'U_enc_loss': loss_enc,
    }

    return U_dict, pred_p, mu1, logvar1, z1, dup_mu1, dup_logvar1, rec_mu1, rec_logvar1

  def cal_U_fixenc(self, U_dict, U_pred_p):
    # L for U
    UL_rec, UL_kld, UL_yprior = U_dict['U_rec'], U_dict['U_kld'], U_dict['U_yprior']

    UL_loss = UL_rec + UL_kld - UL_yprior
    # UL_loss = urec * UL_rec + ukld * torch.abs(ukld_fix - UL_kld) - uyp * UL_yprior
    # U_L_loss = U_rec + min(1.0, self.step / 1000) * U_kld - U_yprior

    # expectation of UL_loss
    U_dict['UL_mean_loss'] = torch.sum((U_pred_p * UL_loss).view(-1, self.label_size), dim=-1)

    # Total U
    # U_loss =  U_L_mean_loss - H
    U_dict['U_loss'] = U_dict['UL_mean_loss'] - U_dict['H']

    # calculate expectations for each term
    U_dict['U_rec'] = torch.sum((U_pred_p * UL_rec).view(-1, self.label_size), dim=-1)
    U_dict['U_kld'] = torch.sum((U_pred_p * UL_kld).view(-1, self.label_size), dim=-1)
    U_dict['U_yprior'] = torch.sum((U_pred_p * UL_yprior).view(-1, self.label_size), dim=-1)
    # fb
    U_dict['U_kld_fb'] = torch.sum((U_pred_p * U_dict['U_kld_fb']).view(-1, self.label_size),
                                   dim=-1)

    # KL(q(y|x) || p(y)) for U
    # bs, label_size
    U_pred_p_rv = U_pred_p.view(-1, self.label_size)
    U_dict['kldy'] = (U_pred_p_rv * (torch.log(U_pred_p_rv + 1e-32) - self.yprior)).mean(dim=1)

    # U_dict['U_loss'] += - U_dict['kldy'] + torch.abs(U_dict['kldy'] - 0.3)

    return U_dict

  def backward_U_fixenc(self, U_dict, cur_bs):
    # backward
    (U_dict['U_loss'].sum() / cur_bs).backward()

    # get sum().item()
    U_dict = {k: v.sum().item() for k, v in U_dict.items()}

    return U_dict

  def get_z1_fixenc(self, lang, batch_in, batch_lens):
    with torch.no_grad():
      # extract feature: mu1, logvar1, eval mode
      self.xlingva.eval()
      mu1, logvar1, hid, loss_dis, loss_enc = self.xlingva.get_gaus(lang, batch_in, batch_lens)

      # stochastic sampling, z for training, mu for eval
      if self.training:
        self.xlingva.inferer.train()
      else:
        self.xlingva.inferer.eval()
      z1 = self.xlingva.inferer.reparameterize(mu1, logvar1)
      # mu debug
      # z1 = mu1
      # mu debug
    return mu1, logvar1, z1, hid, loss_dis, loss_enc

  def get_z1_trainenc(self, lang, batch_in, batch_lens):
    mu1, logvar1, hid, loss_dis, loss_enc = self.xlingva.get_gaus(lang, batch_in, batch_lens)
    z1 = self.xlingva.inferer.reparameterize(mu1, logvar1)
    return mu1, logvar1, z1, hid, loss_dis, loss_enc

  def get_z2(self, z1, batch_ohlb):
    z1y = self.get_z1y(z1, batch_ohlb)
    # z1y -> z2
    mu2, logvar2 = self.z1y_z2(z1y)
    z2 = self.z1y_z2.reparameterize(mu2, logvar2)
    # mu debug
    # z2 = mu2
    # mu debug
    return mu2, logvar2, z2

  def get_z1y_concat(self, z1, batch_ohlb):
    # z1y -> z2
    z1y = torch.cat([z1, batch_ohlb], dim=-1)
    return z1y

  def get_z1y_transconcat(self, z1, batch_ohlb):
    # z1y -> z2
    batch_lb = self.yohtoy(batch_ohlb)
    z1y = torch.cat([z1, batch_lb], dim=-1)
    return z1y

  def get_z1y_transadd(self, z1, batch_ohlb):
    # z1y -> z2
    batch_lb = self.yohtoy_toz2(batch_ohlb)
    z1y = z1 + batch_lb
    if z1y.shape[-1] > 1:
      z1y = self.hbn_z1y(z1y)
    z1y = self.leakyrelu(z1y)
    return z1y

  def get_z1y_gmix(self, z1, batch_ohlb):
    # z1y -> z2
    # concat
    z1y = torch.cat([z1, batch_ohlb], dim=-1)
    '''
    # transadd
    y = self.yohtoy(batch_ohlb)
    z1y = z1 + y
    '''
    return z1y

  def get_z1y_gmix_transadd(self, z1, batch_ohlb):
    # z1y -> z2
    batch_lb = self.yohtoy_toz2(batch_ohlb)
    z1y = z1 + batch_lb
    if z1y.shape[-1] > 1:
      z1y = self.hbn_z1y(z1y)
    z1y = self.leakyrelu(z1y)
    return z1y

  def z2_rec_z1(self, z1, z2, batch_ohlb):
    z2y = self.get_z2y(z2, batch_ohlb)
    # z2y -> z1
    rec_mu1, rec_logvar1 = self.z2y_z1(z2y)
    rec_z1 = self.z2y_z1.reparameterize(rec_mu1, rec_logvar1)

    logpz1 = multi_diag_normal(z1, rec_mu1, rec_logvar1)
    return -logpz1, rec_mu1, rec_logvar1

  def get_z2y_concat(self, z2, batch_ohlb):
    # reconstruction
    z2y = torch.cat([z2, batch_ohlb], dim=-1)
    return z2y

  def get_z2y_transconcat(self, z2, batch_ohlb):
    batch_lb = self.yohtoy(batch_ohlb)
    z2y = torch.cat([z2, batch_lb], dim=-1)
    return z2y

  def get_z2y_transadd(self, z2, batch_ohlb):
    batch_lb = self.yohtoy_toz1(batch_ohlb)
    z2y = z2 + batch_lb
    if z2y.shape[-1] > 1:
      z2y = self.hbn_z2y(z2y)
    z2y = self.leakyrelu(z2y)
    return z2y

  def get_z2y_gmix(self, z2, batch_ohlb):
    return z2

  def get_z2y_gmix_transadd(self, z2, batch_ohlb):
    return z2

  def enumerate_label(self, batch_uin, batch_uohlb):
    # batch_dup_ulens = np.repeat(batch_ulens, repeats = batch_uohlb.shape[1], axis = 0)
    batch_dup_uin = batch_uin.unsqueeze(1).repeat(1, batch_uohlb.shape[1], 1).view(-1,
                                                                                   batch_uin.shape[
                                                                                     1])
    return batch_dup_uin


def calc_au(mus, delta=0.01):
  """compute the number of active units
  """
  if len(mus) == 0:
    return 0, 0

  cnt = 0
  for mean in mus:
    if cnt == 0:
      means_sum = mean.sum(dim=0, keepdim=True)
    else:
      means_sum = means_sum + mean.sum(dim=0, keepdim=True)
    cnt += mean.size(0)

  # (1, nz)
  mean_mean = means_sum / cnt

  cnt = 0
  for mean in mus:
    if cnt == 0:
      var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
    else:
      var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
    cnt += mean.size(0)

  # (nz)
  au_var = var_sum / (cnt - 1)

  return (au_var >= delta).sum().item(), au_var


def get_cyclic_weight(step, step_period):
  # number of steps for increasing
  if (step // step_period) % 2 == 0:
    return 1.0 * (step % step_period) / step_period
  elif (step // step_period) % 2 == 1:
    # number of steps for 1
    return 1.0
