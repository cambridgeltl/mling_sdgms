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

from nn_model.semicldc_model import SEMICLDCModel
from nn_model.xlingva import XlingVA
from nn_model.inferer import Inferer
from nn_model.mlp.cldc_classifier import CLDCClassifier
from utils.logpdfs import multi_diag_normal, cal_kl_gau1, cal_kl_gau2, cal_kl_gau1_fb, cal_kl_gau2_fb


const = np.float128((sympy.log(2 * sympy.pi)).evalf(64))

# fb
l_z1_fb = 10.0
l_z2_fb = 10.0
u_z1_fb = 10.0
u_z2_fb = 30.0

# gen
import pdb
# gen

class AUXSEMICLDCModel(SEMICLDCModel):
  def __init__(self, params, data_list, model_dict=None, task_model_dict=None):
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
    self.cldc_classifier = CLDCClassifier(params, params.aux_cldc_classifier_config)

    # functions
    self.forward = getattr(self, 'forward_{}'.format(params.cldc_train_mode))

    self.get_z1 = getattr(self, 'get_z1_{}'.format(params.cldc_train_mode))

    self.get_z1x = getattr(self, 'get_z1x_{}'.format(params.semicldc_cond_type))
    self.get_z1yx = getattr(self, 'get_z1yx_{}'.format(params.semicldc_cond_type))
    self.get_z2y = getattr(self, 'get_z2y_{}'.format(params.semicldc_cond_type))
    self.get_z2z1y = getattr(self, 'get_z2z1y_{}'.format(params.semicldc_cond_type))
    # calculate kl of z2
    self.kl_z2 = getattr(self, 'kl_z2_{}'.format(params.semicldc_cond_type))

    self.step = 0
    self.anneal_warm_up = params.semicldc_anneal_warm_up
    self.cyclic_period = params.cyclic_period

    # warm up stage
    self.warm_up = False

    self.init_model(task_model_dict)

    self.use_cuda = params.cuda
    if self.use_cuda:
      self.cuda()


  def train_classifier(self, lang, batch_in, batch_lens, batch_lb, training = True):
    # x -> hid_x -> mu1, logva1 -> z1
    mu1, logvar1, z1, x, loss_dis, loss_enc = self.get_z1(lang, batch_in, batch_lens)
    z1x_y = self.get_z1x(z1, x)

    # cldc loss
    cldc_loss, pred_p, pred = self.cldc_classifier(z1x_y, y = batch_lb, training = training)

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


  def init_semicldc_cond_transadd(self, params):
    self.leakyrelu = nn.LeakyReLU()

    # x -> hid for y
    self.xtohid_y = nn.Linear(params.x_hid_dim, params.aux_hid_dim)
    # z1 -> hid for y
    self.z1tohid_y = nn.Linear(params.z_dim, params.aux_hid_dim)
    self.hbn_z1x = nn.BatchNorm1d(params.aux_hid_dim)

    # x -> hid for z2
    self.xtohid_z2 = nn.Linear(params.x_hid_dim, params.aux_hid_dim)
    # z1 -> hid for z2
    self.z1tohid_z2 = nn.Linear(params.z_dim, params.aux_hid_dim)
    # y -> hid for z2
    self.ytohid_z2 = nn.Linear(self.label_size, params.aux_hid_dim)
    # z1,y,x -> z2
    self.hbn_z1xy = nn.BatchNorm1d(params.aux_hid_dim)
    self.z1yx_z2 = Inferer(params, in_dim = params.aux_hid_dim)

    # y -> hid for z1
    self.ytohid_z1 = nn.Linear(self.label_size, params.aux_hid_dim)
    # z2 -> hid for z1
    self.z2tohid_z1 = nn.Linear(params.z_dim, params.aux_hid_dim)
    # y,z2 -> z1
    self.hbn_z2y = nn.BatchNorm1d(params.aux_hid_dim)
    self.z2y_z1 = Inferer(params, in_dim = params.aux_hid_dim)

    # y -> hid for x
    self.ytohid_x = nn.Linear(self.label_size, params.aux_hid_dim)
    # z1 -> hid for x
    self.z1tohid_x = nn.Linear(params.z_dim, params.aux_hid_dim)
    # z2 -> hid for x
    self.z2tohid_x = nn.Linear(params.z_dim, params.aux_hid_dim)
    # y,z2,z1 -> x
    self.hbn_z2z1y = nn.BatchNorm1d(params.aux_hid_dim)
    self.z2z1y_x = nn.Linear(params.aux_hid_dim, params.z_dim)


  def init_semicldc_cond_gmix_transadd(self, params):
    self.init_semicldc_cond_transadd(params)

    # y -> hid for z2
    self.ytohid_z2 = nn.Linear(self.label_size, params.aux_hid_dim)
    # y -> z2
    self.hbn_y = nn.BatchNorm1d(params.aux_hid_dim)
    self.y_z2 = Inferer(params, in_dim = params.aux_hid_dim)


  def forward_L_trainenc_batch(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb):
    L_dict, L_pred, mu1, logvar1, z1, rec_mu1, rec_logvar1, z2 = self.forward_L_fixenc_batch(lang, batch_in, batch_lens, batch_lb, batch_ohlb)

    z2z1y_x = self.get_z2z1y(z2, z1, batch_ohlb)

    # nll_loss
    L_dict['L_nll'] = self.xlingva.decoder(lang, z2z1y_x, batch_in, reduction=None)
    # H(q(z1|x))
    # k/2 + k/2 log(2pi) + 1/2 log(|covariance|)
    L_dict['L_Hz1'] = mu1.shape[1] / 2.0 * (1 + const) + 1 / 2.0 * logvar1.sum(dim=-1)
    # regroup
    L_dict['L_z1kld'] = cal_kl_gau2(mu1, logvar1, rec_mu1, rec_logvar1)
    # fb
    L_dict['L_z1kld_fb'] = cal_kl_gau2_fb(mu1, logvar1, rec_mu1, rec_logvar1, l_z1_fb)

    return L_dict, L_pred


  def forward_U_trainenc_batch(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb):
    U_dict, U_pred_p, mu1, logvar1, dup_z1, dup_mu1, dup_logvar1, rec_mu1, rec_logvar1, z2 = self.forward_U_fixenc_batch(lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb)

    z2z1y_x = self.get_z2z1y(z2, dup_z1, batch_uohlb)

    # nll_loss, expectation
    dup_batch_uin = self.enumerate_label(batch_uin, batch_uohlb)
    U_dict['U_nll'] = self.xlingva.decoder(lang, z2z1y_x, dup_batch_uin, reduction = None)
    U_dict['U_nll'] = (U_dict['U_nll'] * U_pred_p).view(batch_uin.shape[0], -1).sum(dim=1)
    # H(q(z1|x))
    # k/2 + k/2 log(2pi) + 1/2 log(|covariance|)
    U_dict['U_Hz1'] = mu1.shape[1] / 2.0 * (1 + const) + 1 / 2.0 * logvar1.sum(dim=-1)

    # regroup
    U_dict['U_z1kld'] = cal_kl_gau2(dup_mu1, dup_logvar1, rec_mu1, rec_logvar1)
    # fb
    U_dict['U_z1kld_fb'] = cal_kl_gau2_fb(dup_mu1, dup_logvar1, rec_mu1, rec_logvar1, u_z1_fb)
    return U_dict, U_pred_p


  def forward_L_fixenc_batch(self, lang, batch_in, batch_lens, batch_lb, batch_ohlb):
    mu1, logvar1, z1, x, loss_dis, loss_enc = self.get_z1(lang, batch_in, batch_lens)

    z1x_y = self.get_z1x(z1, x)

    # cldc loss for training
    cldc_loss, _, pred = self.cldc_classifier(z1x_y, y = batch_lb, training = True)

    # z1yh -> z2
    mu2, logvar2, z2 = self.get_z2(z1, batch_ohlb, x)

    # au
    self.L_mu1.append(mu1)
    self.L_mu2.append(mu2)

    # z1, z2 distance
    self.L_z1z2kld += 0.5 * torch.mean(
      torch.sum(logvar1 - logvar2 - 1 + ((mu2 - mu1).pow(2) + logvar2.exp()) / logvar1.exp(), dim=1)).item()
    self.L_l2dist += torch.mean(torch.sqrt(torch.sum(((z1 - z2) ** 2), dim=1))).item()
    self.L_cosdist += torch.mean(F.cosine_similarity(z1, z2)).item()

    ''' 
    # gen
    self.eval()
    # fix y sample z
    # 8*4
    y_oh = torch.cat([torch.eye(4), torch.eye(4)]).cuda()
    # 2*4
    mu2 = torch.zeros(2, mu2.shape[1]).cuda()
    logvar2 = torch.zeros(2, mu2.shape[1]).cuda()
    z2 = self.z1yx_z2.reparameterize(mu2, logvar2)
    # 8*4
    z2 = z2.repeat_interleave(4, dim = 0)
    z2y_z1 = self.get_z2y(z2, y_oh)
    # z2y -> z1
    rec_mu1, rec_logvar1 = self.z2y_z1(z2y_z1)
    rec_z1 = self.z2y_z1.reparameterize(rec_mu1, rec_logvar1)
    # z2z1y - > x
    z2z1y_x = self.get_z2z1y(z2, rec_z1, y_oh)

    batch_size = z2z1y_x.shape[0]
    decoded_idxs = []
    # whether each sentence has finished generattion
    finish_idxs = torch.tensor([False] * batch_size).cuda()
    
    lang_idx = self.xlingva.decoder.lang_dict[lang]
    # init hid
    dec_hid = self.xlingva.decoder.z2hid[lang_idx](z2z1y_x).unsqueeze(0)
    # init x, pad
    dec_in = torch.zeros(dec_hid.shape[1], dtype = torch.long).unsqueeze(1).cuda()
    # max length
    for di in range(200):
      embedded = self.xlingva.decoder.embeddings.embeddings[lang_idx](dec_in)
      batch_size = embedded.shape[0]
      # concatenate with z
      embedded = torch.cat((embedded, z2z1y_x.unsqueeze(1)), dim = -1)
      # linear transformation
      embedded = self.xlingva.decoder.zx2decin[lang_idx](embedded)
      out, dec_hid = self.xlingva.decoder.rnns[lang_idx](embedded, dec_hid)
      scores = self.xlingva.decoder.hid2vocab[lang_idx](out).squeeze(1)
      scores = F.log_softmax(scores, dim = 1)
      log_prob, topi = scores.data.topk(1)
      decoded_idx = topi[:, 0].detach()
      decoded_idxs.append(decoded_idx.unsqueeze(1))
      finish_idxs = (finish_idxs | (decoded_idx == 0))

      if finish_idxs.all().item():
        break
      
      dec_in = topi[:, 0].detach().unsqueeze(1)  # detach from history as input 
    
    decoded_idxs = torch.stack(decoded_idxs, dim = -1).cpu().squeeze(1).numpy()
    sents = []
    for i in range(decoded_idxs.shape[0]):
      sent = []
      for j in range(decoded_idxs.shape[1]):
        sent.append(self.xlingva.embeddings.embeddings[lang_idx].vocab.idx2word[decoded_idxs[i][j]])
      sents.append(' '.join(sent))
    pdb.set_trace()
    # gen
    ''' 

    # reconstruct z1 from z2
    rec_loss, rec_mu1, rec_logvar1 = self.z2_rec_z1(z1, z2, batch_ohlb)
    # kl divergence of z2
    kld = self.kl_z2(mu2, logvar2, batch_ohlb)
    # get y prior
    yprior = batch_ohlb @ self.yprior

    # fb
    kld_fb = cal_kl_gau1_fb(mu2, logvar2, l_z2_fb)

    L_dict = {
      'L_rec': rec_loss,
      'L_kld': kld,
      'L_yprior': yprior,
      'L_cldc_loss': cldc_loss,
      'L_kld_fb': kld_fb,
      'L_dis_loss': loss_dis,
      'L_enc_loss': loss_enc,
    }
    return L_dict, pred, mu1, logvar1, z1, rec_mu1, rec_logvar1, z2


  def forward_U_fixenc_batch(self, lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb):
    mu1, logvar1, z1, x, loss_dis, loss_enc = self.get_z1(lang, batch_uin, batch_ulens)

    z1x_y = self.get_z1x(z1, x)

    # use classifier to infer
    _, pred_p, _ = self.cldc_classifier(z1x_y, y = None, training = True)

    # duplicate z1, enumerate y
    # bs * label_size, z_dim
    dup_z1 = self.enumerate_label(z1, batch_uohlb)
    dup_mu1 = self.enumerate_label(mu1, batch_uohlb)
    dup_logvar1 = self.enumerate_label(logvar1, batch_uohlb)
    dup_x = self.enumerate_label(x, batch_uohlb)

    mu2, logvar2, z2 = self.get_z2(dup_z1, batch_uohlb, dup_x)

    # au
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

    #fb
    kld_fb = cal_kl_gau1_fb(mu2, logvar2, u_z2_fb)

    #fb
    U_dict = {
      'U_rec': rec_loss,
      'U_kld': kld,
      'U_yprior': yprior,
      'H': H,
      'U_kld_fb': kld_fb,
      'U_dis_loss': loss_dis,
      'U_enc_loss': loss_enc,
    }
    return U_dict, pred_p, mu1, logvar1, dup_z1, dup_mu1, dup_logvar1, rec_mu1, rec_logvar1, z2


  def get_z1x_transadd(self, z1, x):
    z1_y = self.z1tohid_y(z1)
    x_y = self.xtohid_y(x)
    z1x_y = z1_y + x_y
    if z1x_y.shape[0] > 1:
      z1x_y = self.hbn_z1x(z1x_y)
    z1x_y = self.leakyrelu(z1x_y)
    return z1x_y


  def get_z1x_gmix_transadd(self, z1, x):
    return self.get_z1x_transadd(z1, x)


  def get_z2(self, z1, batch_ohlb, x):
    z1yx_z2 = self.get_z1yx(z1, batch_ohlb, x)
    # z1y -> z2
    mu2, logvar2 = self.z1yx_z2(z1yx_z2)
    z2 = self.z1yx_z2.reparameterize(mu2, logvar2)
    # mu debug
    # z2 = mu2
    # mu debug
    return mu2, logvar2, z2


  def get_z1yx_transadd(self, z1, batch_ohlb, x):
    # z1yx -> z2
    y_z2 = self.ytohid_z2(batch_ohlb)
    z1_z2 = self.z1tohid_z2(z1)
    x_z2 = self.xtohid_z2(x)
    z1yx_z2 = z1_z2 + y_z2 + x_z2
    if z1yx_z2.shape[0] > 1:
      z1yx_z2 = self.hbn_z1xy(z1yx_z2)
    z1yx_z2 = self.leakyrelu(z1yx_z2)
    return z1yx_z2


  def get_z1yx_gmix_transadd(self, z1, batch_ohlb, x):
    return self.get_z1yx_transadd(z1, batch_ohlb, x)


  def z2_rec_z1(self, z1, z2, batch_ohlb):
    z2y_z1 = self.get_z2y(z2, batch_ohlb)
    # z2y -> z1
    rec_mu1, rec_logvar1 = self.z2y_z1(z2y_z1)
    rec_z1 = self.z2y_z1.reparameterize(rec_mu1, rec_logvar1)

    logpz1 = multi_diag_normal(z1, rec_mu1, rec_logvar1)
    return -logpz1, rec_mu1, rec_logvar1


  def get_z2y_transadd(self, z2, batch_ohlb):
    y_z1 = self.ytohid_z1(batch_ohlb)
    z2_z1 = self.z2tohid_z1(z2)
    # p(z1|z2, y)
    z2y_z1 = z2_z1 + y_z1
    # p(z1|z2)
    #z2y_z1 = z2_z1
    if z2y_z1.shape[0] > 1:
      z2y_z1 = self.hbn_z2y(z2y_z1)
    z2y_z1 = self.leakyrelu(z2y_z1)
    return z2y_z1


  def get_z2y_gmix_transadd(self, z2, batch_ohlb):
    return self.get_z2y_transadd(z2, batch_ohlb)


  def get_z2z1y_transadd(self, z2, z1, batch_ohlb):
    z2_x = self.z2tohid_x(z2)
    z1_x = self.z1tohid_x(z1)
    y_x = self.ytohid_x(batch_ohlb)
    z2z1y_x = z2_x + z1_x + y_x
    if z2z1y_x.shape[0] > 1:
      z2z1y_x = self.hbn_z2z1y(z2z1y_x)
    z2z1y_x = self.leakyrelu(z2z1y_x)
    z2z1y_x = self.z2z1y_x(z2z1y_x)
    return z2z1y_x


  def get_z2z1y_gmix_transadd(self, z2, z1, batch_ohlb):
    return self.get_z2z1y_transadd(z2, z1, batch_ohlb)


  def kl_z2_gmix_transadd(self, mu_post, logvar_post, batch_ohlb):
    y_z2 = self.ytohid_z2(batch_ohlb)
    if y_z2.shape[0] > 1:
      y_z2 = self.hbn_y(y_z2)
    mu_prior, logvar_prior = self.y_z2(y_z2)
    kld = cal_kl_gau2(mu_post, logvar_post, mu_prior, logvar_prior)
    return kld
