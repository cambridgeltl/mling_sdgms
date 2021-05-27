# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Args for vae_cross_emb
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import torch


def create_args():
  parser = argparse.ArgumentParser(description = 'VAE cross-lingual learning')
  
  # languages for experiments
  parser.add_argument("--langs", nargs = '+', default = ['en_de'],
      help = "language list")

  # segmentation model
  parser.add_argument("--seg_model_path",
                      default = '/yourpath/data/europarl/sw/ende/spm_1e4/ende.spm.model',
                      #default = None,
                      help = "segmentation model path")

  # vocabulary path
  parser.add_argument("--vocab_path", nargs = '+',
      default = ['/yourpath/data/europarl/sw/ende/spm_1e4/ende.spm.vocab'],
      #default = ['/yourpath/data/europarl/europarl-v7.de-en.en.preproc.train.dict',
                 #'/yourpath/data/europarl/europarl-v7.de-en.de.preproc.train.dict'],
      #default = ['/yourpath/data/europarl/europarl-v7.fr-en.en.preproc.train.dict',
                 #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.train.dict'],
      #default = ['/yourpath/data/europarl/europarl-v7.de-en.de.preproc.train.dict',
                 #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.train.dict'],
      help = "vocabulary path, sorted according to language list")
  # pretrained embedding file paths
  parser.add_argument("--pretrained_emb_path", nargs = '+',
      default = [None],
      #default = [None,
                 #None],
      #default = ['/yourpath/train.1.0_per.50_pre.n_emb.300_ls.xlnllankl_adv.3ops.1hfunc.fixa0.100_enc.bilstm_sp.1_bs.128_op.Adam_lr.0.0005/en.train.1.0_per.50_pre.n_emb.300_ls.xlnllankl_adv.3ops.1hfunc.fixa0.100_enc.bilstm_sp.1_bs.128_op.Adam_lr.0.0005.txt',
                 #'/yourpath/train.1.0_per.50_pre.n_emb.300_ls.xlnllankl_adv.3ops.1hfunc.fixa0.100_enc.bilstm_sp.1_bs.128_op.Adam_lr.0.0005/de.train.1.0_per.50_pre.n_emb.300_ls.xlnllankl_adv.3ops.1hfunc.fixa0.100_enc.bilstm_sp.1_bs.128_op.Adam_lr.0.0005.txt'],
      #default = ['/yourpath/europarl-v7.de-en.en.preproc.train.muse.txt',
                 #'/yourpath/europarl-v7.de-en.de.preproc.train.muse.txt',],
      #default = [
                 #'/yourpath/europarl-v7.de-en.en.preproc.train.ft.txt',
                 #'/yourpath/europarl-v7.de-en.de.preproc.train.ft.txt',
                 #],
      #default = [
                #'/yourpath/original_code/out.en.txt',
                #'/yourpath/original_code/out.de.txt',
                #],
      help = "pretrained embedding path, sorted according to language list")

  #--------------------------------------------------     
  # Data for pretraining
  #--------------------------------------------------      
  # pretraining input data
  parser.add_argument("--train_path", nargs = '+',
      default = ['/yourpath/data/europarl/europarl-v7.de-en.ende.preproc.train'],
      #default = ['/yourpath/data/europarl/europarl-v7.de-en.en.preproc.train',
                 #'/yourpath/data/europarl/europarl-v7.de-en.de.preproc.train'],
      #default = ['/yourpath/data/europarl/europarl-v7.fr-en.en.preproc.train',
                 #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.train'],
      #default=['/yourpath/data/europarl/europarl-v7.de-en.de.preproc.train',
                #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.train'],
                      help = "pretraining train path, sorted according to language list")

  parser.add_argument("--dev_path", nargs = '+',
      default = ['/yourpath/data/europarl/europarl-v7.de-en.ende.preproc.dev'],
      #default = ['/yourpath/data/europarl/europarl-v7.de-en.en.preproc.dev',
                 #'/yourpath/data/europarl/europarl-v7.de-en.de.preproc.dev'],
      #default = ['/yourpath/data/europarl/europarl-v7.fr-en.en.preproc.dev',
                 #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.dev'],
      #default = ['/yourpath/data/europarl/europarl-v7.de-en.de.preproc.dev',
                 #'/yourpath/data/europarl/europarl-v7.fr-en.fr.preproc.dev'],
      help = "pretraining dev path, sorted according to language list")

  #--------------------------------------------------     
  # CLDC data 
  #--------------------------------------------------
  # languages for experiments
  parser.add_argument("--cldc_data_langs", nargs = '+', default = ['en', 'de'],
      help = "language list of data for cldc task")

  parser.add_argument("--cldc_path", nargs = '+',
      #default = ['/yourpath/data/cldc/cldc.en.pth',
                 #'/yourpath/data/cldc/cldc.de.pth'],
      default = ['/yourpath/data/mldoc/en/mldoc.en.pth',
                 '/yourpath/data/mldoc/de/mldoc.de.pth'],
      #default = ['/yourpath/data/mldoc/en/mldoc.en.pth',
                 #'/yourpath/data/mldoc/fr/mldoc.fr.pth'],
      help = "CLDC file path, sorted according to language list")

  #--------------------------------------------------     
  # MLSA data 
  #--------------------------------------------------      
  parser.add_argument("--mlsa_path", nargs = '+', 
      default = ['/yourpath/data/mlsa/mlsa.en.books.pth',
                 '/yourpath/data/mlsa/mlsa.de.books.pth'],
      help = "mlsa file path, sorted according to language list")

  # save model
  parser.add_argument("--save_model", type = str, default = "out",
      help = "save current model path")
  # load model
  parser.add_argument("--load_model", type = str, default = None,
      help = "model file to be loaded")
  
  # cuda
  parser.add_argument("--cuda", action = 'store_true', default = False, 
      help = "enable cuda")

  args = parser.parse_args()
  args.cuda = args.cuda if torch.cuda.is_available() else False
  return args
