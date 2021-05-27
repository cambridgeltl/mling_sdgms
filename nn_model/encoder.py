# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Encoder of vae cross-lingual emb
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pdb


class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.enc_type = params.enc_type
    self.in_dim = params.enc_in_dim
    self.hid_dim = params.enc_hid_dim
    self.num_layers = params.enc_num_layers
    self.bidi = params.enc_bidi

    if self.enc_type == 'lstm':
      self.rnn = nn.LSTM(self.in_dim,
                         self.hid_dim,
                         num_layers = self.num_layers,
                         bidirectional = self.bidi,
                         dropout = params.enc_do,
                         batch_first = True)
    elif self.enc_type == 'gru':
      self.rnn = nn.GRU(self.in_dim,
                        self.hid_dim,
                        num_layers = self.num_layers,
                        bidirectional = self.bidi,
                        dropout = params.enc_do,
                        batch_first = True)

    self.use_cuda = params.cuda


  def forward(self, input_word_embs, batch_seq_lens):
    if self.enc_type == 'lstm':
      #out, (hn, cn) = self.rnn(input_word_embs)

      # lead to different results
      packed_input_word_embs = pack_padded_sequence(input_word_embs, batch_seq_lens, batch_first = True, enforce_sorted = False)
      packed_out, (hn, cn) = self.rnn(packed_input_word_embs)
      out, _ = pad_packed_sequence(packed_out, batch_first = True)
    elif self.enc_type == 'gru':
      #out, hn = self.rnn(input_word_embs)

      # lead to different results
      packed_input_word_embs = pack_padded_sequence(input_word_embs, batch_seq_lens, batch_first = True, enforce_sorted = False)
      packed_out, hn = self.rnn(packed_input_word_embs)
      out, _ = pad_packed_sequence(packed_out, batch_first = True)

    # last hidden state
    last_hid = torch.cat([hn[0:4:2], hn[1:4:2]], 2)[-1]
    return last_hid
