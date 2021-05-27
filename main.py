# -*- coding: UTF-8 -*-
#!/usr/bin/python3

#************************************************************
# Imported Libraries
#************************************************************
import os
import datetime

import torch
import random

from args import create_args
from params import Params
from data_model.vocab_reader import VocabReader, build_inter_vocab_mapping
from data_model.sw_vocab_reader import SPMVocabReader
import train_parallel, train_xling, train_cldc, train_semicldc, train_xlsemicldc, train_mlsa
from utils.legacy import VocabReader_compat

import pdb

# For fixing seeds
seed = 1234
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def main(params):
  # load pretrained model
  vocab_dict, model_dict, task_model_dict = get_pretrained_model(params)
  # get vocabs for different languages
  vocabs = get_vocabs(params, vocab_dict)

  # general info
  print_info(params, vocabs)
  
  if params.task == 'xl' or params.task == 'xl-adv':
    train_xling.main(params, vocabs, model_dict)
  elif params.task == 'cldc':
    train_cldc.main(params, vocabs, model_dict)
  elif params.task == 'semi-cldc':
    train_semicldc.main(params, vocabs, model_dict)
  elif params.task == 'aux-semi-cldc':
    train_semicldc.main(params, vocabs, model_dict = model_dict, task_model_dict = task_model_dict, aux = True)
  elif params.task == 'xl-semi-cldc':
    train_xlsemicldc.main(params, vocabs, model_dict)
  elif params.task == 'aux-xl-semi-cldc':
    train_xlsemicldc.main(params, vocabs, model_dict, aux = True)
  elif params.task == 'mlsa':
    train_mlsa.main(params, vocabs, model_dict)

  # specific info
  #params.emb_out_path = gen_emb_out_path(params)

  print('\nTraining finished.')


def get_pretrained_model(params):
  vocab_dict = {}
  if params.load_model is not None:
    # pretrained model
    pretrained_model = torch.load(params.load_model)
    if 'model' not in pretrained_model:
      # task-specific model
      return vocab_dict, None, pretrained_model
    else:
      # old model compatability
      if 'vocab_x' in pretrained_model or 'vocab_y' in pretrained_model:
        if 'vocab_x' in pretrained_model:
          vocab_dict['en'] = VocabReader_compat(VocabReader(params, 'en'), pretrained_model['vocab_x'])
        if 'vocab_y' in pretrained_model:
          vocab_dict['de'] = VocabReader_compat(VocabReader(params, 'de'), pretrained_model['vocab_y'])
      else:
        for lang in params.langs:
          if params.seg_type == 'word':
            vocab_dict[lang] = VocabReader_compat(VocabReader(params, lang), pretrained_model[lang])
          elif params.seg_type == 'spm':
            vocab_dict[lang] = VocabReader_compat(SPMVocabReader(params, lang), pretrained_model[lang])
      model_dict = pretrained_model['model']
  else:
    model_dict = None
  return vocab_dict, model_dict, None


def get_vocabs(params, vocab_dict):
  vocabs = []
  for lang in params.langs:
    if lang in vocab_dict:
      vocabs.append(vocab_dict[lang])
    else:
      if params.seg_type == 'word':
        vocabs.append(VocabReader(params, lang))
      elif params.seg_type == 'spm':
        vocabs.append(SPMVocabReader(params, lang))
  #build_inter_vocab_mapping(vocabs)
  return vocabs


def print_info(params, vocabs):
  params.log_path = params.log_path + '_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now()) if params.log_path is not None else None  
  info = ('{}\n'.format('=' * 80) + 
          'Langs: {}\n'.format(' '.join(params.langs)) +  
          'Vocab size: {}\n'.format(' '.join([str(vocab.vocab_size) for vocab in vocabs])) +
          'Pretrained model: \n{}\n'.format(params.load_model if params.load_model is not None else 'None') + 
          'Word embedding dim: {}\n'.format(params.emb_dim) +
          'Pretrained word embeddings: \n{}\n'.format('\n'.join([ep if ep is not None else 'None' for ep in params.pretrained_emb_path])) + 
          'CUDA: {}\n'.format(params.cuda) + 
          'Log file path: {}\n'.format(params.log_path if params.log_path is not None else 'None') + 
          '{}\n'.format('=' * 80) +
          'Task: {}'.format(params.task)
         )
  print(info)
  

def gen_emb_out_path(params):
  cur_time = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
  # annealing str
  if params.an == 'sigmoid':
    # sigmoid annealing
    an_str = params.an + 'ep' + str(params.sigmoid_x0ep) + 'k' + str(params.sigmoid_k)
  elif params.an == 'beta':
    # beta annealing, linear
    an_str = 'betalinear'
  elif params.an == 'fixa':
    an_str = params.an + '{:.3f}'.format(params.fixed_alpha)
  elif params.an == 'standard' or params.an == 'nokld':
    # standard vae or not optimize kld
    an_str = params.an

  # early stopping str
  es_str = '{},{}'.format(params.patience, params.min_delta)

  if params.task_type == 'pa' or params.task_type == 'xl' or params.task_type == 'mo':
    mono_type = params.mono_type if params.task_type == 'mo' else ''
    out_str = (
               'train.{}_'.format(1.0 / params.train_scale) + 
               'per.{}_'.format(params.corpus_percentile) + 
               'pre.{}_'.format('n' if params.pretrained is None else params.pretrained) + 
               'emb.{}_'.format(params.emb_dim) + 
               'ls.{}_'.format(mono_type + params.corpus_type + params.ls_type) + 
               ('adv.3ops.1h' if params.adv_training else '') + 
               'func.{}_'.format(an_str) + 
               'enc.{}_'.format(('bi' if params.enc_bidi else 'uni') + params.enc_type) + 
               'sp.{}_'.format(params.sample_n) + 
               'bs.{}_'.format(params.bs) + 
               'op.{}_'.format('Adam') + 
               'lr.{}'.format(params.init_learning_rate)
              )
  elif params.task_type == 'cldc' or params.task_type == 'semi-cldc':
    # get the pretrained model prefix
    if params.load_model is None:
      pretrained_prefix = '' 
    else:
      pretrained_prefix = os.path.basename(params.load_model)
      pretrained_prefix = '_' + pretrained_prefix[: pretrained_prefix.find('.pth')]
    mlp_str = params.cldc_mlp_type if params.cldc_mlp_type == 'sl' else (params.cldc_mlp_type + params.activation + str(params.cldc_do_rate))
    cldc_str = ('{}red.{}_'.format(params.task_type, params.cldc_train_mode) + 
                ('{}_'.format(params.semicldc_yprior_type) if params.task_type == 'semi-cldc' else '' ) +
                ('{}_'.format(params.semicldc_cond_type) if params.task_type == 'semi-cldc' else '' ) +
                'train.{}_'.format(params.cldc_train_scale) + 
                'val.{}_'.format(params.CLDC_VAL_EVERY) + 
                '{}_'.format('2'.join(params.cldc_part)) + 
                '{}_'.format(mlp_str) + 
                'lr.{}_'.format(params.cldc_init_learning_rate) + 
                'per.{}_'.format(params.cldc_percentile) + 
                'pre.{}'.format('n' if params.pretrained is None else params.pretrained)
                )
    out_str = cldc_str + pretrained_prefix + '_' + cur_time
  return out_str


if __name__ == '__main__':
  arguments = create_args()
  params = Params(arguments)
  main(params)
