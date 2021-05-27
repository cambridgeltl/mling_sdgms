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

from nn_model.aux_semicldc_model import AUXSEMICLDCModel
from nn_model.semicldc_model import calc_au


class AUXXLSEMICLDCModel(AUXSEMICLDCModel):
  def __init__(self, params, data_list, model_dict=None):
    AUXSEMICLDCModel.__init__(self, params, data_list, model_dict = model_dict)

    self.src_lang, self.trg_lang = params.cldc_langs
    # supervised and unsupervised loss hyperparameters for both languages
    self.src_le, self.src_ue, self.trg_le, self.trg_ue = params.src_le, params.src_ue, params.trg_le, params.trg_ue
    # classification loss alpha
    self.src_cls_alpha, self.trg_cls_alpha = params.src_cls_alpha, params.trg_cls_alpha
    self.adv_training = params.adv_training


  def forward_trainenc(self, src_batch_in, src_batch_lens, src_batch_lb, src_batch_ohlb,
                       src_batch_uin, src_batch_ulens, src_batch_ulb, src_batch_uohlb,
                       trg_batch_in, trg_batch_lens, trg_batch_lb, trg_batch_ohlb,
                       trg_batch_uin, trg_batch_ulens, trg_batch_ulb, trg_batch_uohlb):
    # warm up
    if self.warm_up:
      src_loss_dict, src_L_pred = self.train_classifier(self.src_lang, src_batch_in, src_batch_lens, src_batch_lb)

      trg_loss_dict = defaultdict(float)
      trg_L_pred = None
      if self.trg_le > .0 and trg_batch_in is not None:
        trg_loss_dict, trg_L_pred = self.train_classifier(self.trg_lang, trg_batch_in, trg_batch_lens, trg_batch_lb)

      return src_loss_dict, trg_loss_dict, src_L_pred, trg_L_pred

    # src
    src_loss_dict, src_L_pred = self.forward_trainenc_onelang(src_batch_in, src_batch_lens, src_batch_lb, src_batch_ohlb,
                                                              src_batch_uin, src_batch_ulens, src_batch_ulb, src_batch_uohlb,
                                                              lang = self.src_lang, le = self.src_le, ue = self.src_ue, cls_alpha = self.src_cls_alpha)

    # trg
    trg_loss_dict, trg_L_pred = self.forward_trainenc_onelang(trg_batch_in, trg_batch_lens, trg_batch_lb, trg_batch_ohlb,
                                                              trg_batch_uin, trg_batch_ulens, trg_batch_ulb, trg_batch_uohlb,
                                                              lang = self.trg_lang, le = self.trg_le, ue = self.trg_ue, cls_alpha = self.trg_cls_alpha)

    self.step += 1

    return src_loss_dict, trg_loss_dict, src_L_pred, trg_L_pred



  def forward_trainenc_onelang(self, batch_in, batch_lens, batch_lb, batch_ohlb,
                               batch_uin, batch_ulens, batch_ulb, batch_uohlb,
                               lang, le, ue, cls_alpha):
    # au
    self.L_mu1, self.U_mu1 = [], []
    self.L_mu2, self.U_mu2 = [], []
    # z1, z2 distance
    self.L_l2dist, self.L_cosdist, self.L_z1z2kld = .0, .0, .0
    self.U_l2dist, self.U_cosdist, self.U_z1z2kld = .0, .0, .0

    L_dict, U_dict = defaultdict(), defaultdict()
    L_pred = None

    # calculate L loss
    if le > .0 and batch_in is not None:
      L_dict, L_pred = self.forward_L_trainenc(lang, batch_in, batch_lens, batch_lb, batch_ohlb,
                                               le = le, cls_alpha = cls_alpha)

    # calculate U loss
    if ue > .0 and batch_uin is not None:
      U_dict = self.forward_U_trainenc(lang, batch_uin, batch_ulens, batch_ulb, batch_uohlb, ue = ue)

    # merge two dicts
    loss_dict = defaultdict(float, {**L_dict, **U_dict})

    # z1, z2 distance
    loss_dict['L_l2_dist'], loss_dict['L_cosdist'], loss_dict['L_z1z2kld'] = self.L_l2dist, self.L_cosdist, self.L_z1z2kld
    loss_dict['U_l2_dist'], loss_dict['U_cosdist'], loss_dict['U_z1z2kld'] = self.U_l2dist, self.U_cosdist, self.U_z1z2kld

    # total MEAN loss
    loss_dict['total_loss'] = loss_dict['L_loss_trainenc'] + loss_dict['U_loss_trainenc'] + cls_alpha * loss_dict['L_cldc_loss']

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
