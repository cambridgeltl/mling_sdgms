# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Old model compatability
"""

#************************************************************
# Imported Libraries
#************************************************************

xlingva_param_map = {
                       'encoder.embeddings.0.weight':         'embeddings.embeddings.0.embeddings.weight', 
                       'encoder.embeddings.1.weight':         'embeddings.embeddings.1.embeddings.weight', 
                       'encoder.rnn.weight_ih_l0':            'encoder.rnn.weight_ih_l0', 
                       'encoder.rnn.weight_hh_l0':            'encoder.rnn.weight_hh_l0', 
                       'encoder.rnn.bias_ih_l0':              'encoder.rnn.bias_ih_l0', 
                       'encoder.rnn.bias_hh_l0':              'encoder.rnn.bias_hh_l0', 
                       'encoder.rnn.weight_ih_l0_reverse':    'encoder.rnn.weight_ih_l0_reverse', 
                       'encoder.rnn.weight_hh_l0_reverse':    'encoder.rnn.weight_hh_l0_reverse', 
                       'encoder.rnn.bias_ih_l0_reverse':      'encoder.rnn.bias_ih_l0_reverse', 
                       'encoder.rnn.bias_hh_l0_reverse':      'encoder.rnn.bias_hh_l0_reverse', 
                       'encoder.rnn.weight_ih_l1':            'encoder.rnn.weight_ih_l1', 
                       'encoder.rnn.weight_hh_l1':            'encoder.rnn.weight_hh_l1', 
                       'encoder.rnn.bias_ih_l1':              'encoder.rnn.bias_ih_l1', 
                       'encoder.rnn.bias_hh_l1':              'encoder.rnn.bias_hh_l1', 
                       'encoder.rnn.weight_ih_l1_reverse':    'encoder.rnn.weight_ih_l1_reverse', 
                       'encoder.rnn.weight_hh_l1_reverse':    'encoder.rnn.weight_hh_l1_reverse', 
                       'encoder.rnn.bias_ih_l1_reverse':      'encoder.rnn.bias_ih_l1_reverse', 
                       'encoder.rnn.bias_hh_l1_reverse':      'encoder.rnn.bias_hh_l1_reverse', 
                       'discriminator.mlp.0.weight':          'discriminator.mlp.mlp.0.weight',
                       'discriminator.mlp.0.bias':            'discriminator.mlp.mlp.0.bias', 
                       'discriminator.mlp.2.weight':          'discriminator.mlp.mlp.2.weight', 
                       'discriminator.mlp.2.bias':            'discriminator.mlp.mlp.2.bias', 
                       'inferer.i2h.weight':                  'inferer.i2h.weight', 
                       'inferer.i2h.bias':                    'inferer.i2h.bias', 
                       'inferer.hbn.weight':                  'inferer.hbn.weight', 
                       'inferer.hbn.bias':                    'inferer.hbn.bias', 
                       'inferer.hbn.running_mean':            'inferer.hbn.running_mean', 
                       'inferer.hbn.running_var':             'inferer.hbn.running_var', 
                       'inferer.hbn.num_batches_tracked':     'inferer.hbn.num_batches_tracked', 
                       'inferer.mu.weight':                   'inferer.mu.weight', 
                       'inferer.mu.bias':                     'inferer.mu.bias', 
                       'inferer.logvar.weight':               'inferer.logvar.weight', 
                       'inferer.logvar.bias':                 'inferer.logvar.bias', 
                       'decoder.embeddings.0.weight':         'decoder.embeddings.embeddings.0.embeddings.weight', 
                       'decoder.embeddings.1.weight':         'decoder.embeddings.embeddings.1.embeddings.weight', 
                       'decoder.biases.0':                    'decoder.biases.0', 
                       'decoder.biases.1':                    'decoder.biases.1', 
                       }


def VocabReader_compat(cur_vocab, pre_vocab):
  for k in vars(cur_vocab).keys():
    setattr(cur_vocab, k, getattr(pre_vocab, k))
  return cur_vocab
