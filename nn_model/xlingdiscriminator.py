# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Language Discriminator
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn as nn

import pdb


class XlingDiscriminator(nn.Module):
  def __init__(self, params):
    super(XlingDiscriminator, self).__init__()
    bidi = 2 if params.enc_bidi else 1
    self.in_emb = bidi * params.enc_hid_dim
    self.hid_emb = 1024
    
    self.mlp = nn.Sequential(
        nn.Linear(self.in_emb, self.hid_emb),
        nn.LeakyReLU(),
        #nn.Linear(self.hid_emb, self.hid_emb),
        #nn.LeakyReLU(),
        nn.Linear(self.hid_emb, 1)
        )

    self.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')


  def forward(self, lang, hid):
    o = self.mlp(hid)

    # 1 for en, 0 for not en
    dis_label = torch.ones_like(o) if lang == 'en' else torch.zeros_like(o)
    enc_label = 1 - dis_label
    loss_dis = self.criterion(o, dis_label)
    loss_enc = self.criterion(o, enc_label)

    return loss_dis, loss_enc

