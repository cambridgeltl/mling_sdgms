# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Preprocess text
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import re
import regex
from tqdm import tqdm
import random
import os
import torch
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET

from mosestokenizer import MosesTokenizer

random.seed(1234)

import pdb


def create_args():
  parser = argparse.ArgumentParser(description = 'Preprocessing text')

  # input file prefix
  parser.add_argument("--input", type = str,
      default = '/yourpath/data/europarl/europarl-v7.fr-en',
      help = "input data path")

  # sample n sentences for dev
  parser.add_argument("--n_sample", type = int, 
      default = 13995,
      help = "input data path")

  # cldc
  parser.add_argument("--cldc_input", type = str,
      default = '/yourpath/data/cldc/',
      help = "cldc input")
  parser.add_argument("--cldc_langs", nargs = '+',
      default = ['en', 'de'],
      help = "cldc langs")

  # mldoc
  parser.add_argument("--mldoc_input", type = str,
      default = '/yourpath/data/mldoc',
      help = "mldoc input")
  parser.add_argument("--mldoc_langs", nargs = '+',
      default = ['en', 'de', 'fr', 'it', 'es', 'ru', 'zh', 'ja'],
      help = "mldoc langs")

  # mlsa
  parser.add_argument("--mlsa_input", type = str,
      default = '/yourpath/data/clsa/cls-acl10-unprocessed/',
      help = "mlsa input")
  parser.add_argument("--mlsa_langs", nargs = '+',
      default = ['en', 'de', 'fr', 'jp'],
      help = "mlsa langs")
  parser.add_argument("--mlsa_domains", nargs = '+',
      default = ['books', 'dvd', 'music'],
      help = "mlsa domains")

  args = parser.parse_args()
  return args


def preproc_europarl(args):
  """
    - tokenization
    - lower case
    - sub digit with 0
    - remove all punctuations
    - remove redundant spaces and emtpy lines
    - (optional) cut long sentences to a reasonable length
  """
  langs = args.input[args.input.rfind('.') + 1:].strip().split('-')
  # only 2 languages
  assert(len(langs) == 2)
  lang1, lang2 = langs
  tokenizer1 = MosesTokenizer(lang1)
  tokenizer2 = MosesTokenizer(lang2)
  # read corpus
  with open(args.input + '.{}'.format(lang1), 'r') as fin1, \
       open(args.input + '.{}'.format(lang2), 'r') as fin2:
    text1 = fin1.readlines()
    text2 = fin2.readlines()
  assert(len(text1) == len(text2))
  
  with open(args.input + '.{}.preproc'.format(lang1), 'w') as fout1, \
       open(args.input + '.{}.preproc'.format(lang2), 'w') as fout2:
    for i, line1 in tqdm(enumerate(text1), total = len(text1)):
      # each line is a sentence
      line1 = line1.strip()
      line2 = text2[i].strip()
    
      line1 = preproc_text(line1, tokenizer1)
      line2 = preproc_text(line2, tokenizer2)

      # remove emtpy lines
      if not line1 or not line2:
        continue

      fout1.write(line1 + '\n')
      fout2.write(line2 + '\n')


def preproc_text(line, tokenizer):
    # tokenization
    line = ' '.join(tokenizer(line))
    # lower case
    line = line.lower() 
    # substitute digits with 0
    line = re.sub("\d", "0", line)      
    # remove all punctuations of unicode format
    line = regex.sub(r"\p{P}+", '', line)
    # remove reduandent spaces
    linevec = [w.strip() for w in line.split(' ') if w.strip()]
    # cut long sentence
    #linevec = linevec[:51]
    line = ' '.join(linevec)
    return line


def gen_dev(args):
  """
  generate dev dataset
  """
  with open(args.input + '.en.preproc', 'r') as fin1, \
       open(args.input + '.fr.preproc', 'r') as fin2:
    en_lines = fin1.readlines()
    de_lines = fin2.readlines()
  assert(len(en_lines) == len(de_lines))
  line_n = len(en_lines)
  dev_idxs = dict(zip(random.sample(range(line_n), args.n_sample), [1] * args.n_sample)) 

  en_train_lines = []
  en_dev_lines = []
  de_train_lines = []
  de_dev_lines = []
  for i in range(line_n):
    if i in dev_idxs:
      de_dev_lines.append(de_lines[i])
      en_dev_lines.append(en_lines[i])
    else:
      de_train_lines.append(de_lines[i])
      en_train_lines.append(en_lines[i])
  assert(len(de_dev_lines) == len(en_dev_lines) and len(de_dev_lines) == args.n_sample)
  with open(args.input + '.en.preproc.train', 'w') as fout1, \
       open(args.input + '.fr.preproc.train', 'w') as fout2, \
       open(args.input + '.en.preproc.dev', 'w') as fout3, \
       open(args.input + '.fr.preproc.dev', 'w') as fout4:
         fout1.write(''.join(en_train_lines))
         fout2.write(''.join(de_train_lines))
         fout3.write(''.join(en_dev_lines))
         fout4.write(''.join(de_dev_lines))


def gen_cldc_pth(args):
  for lang in args.cldc_langs:
    input_texts = {}
    train_texts = {}
    dev_texts = {}
    test_texts = {}
    cldc_inputs = [os.path.join(args.cldc_input, 'train', '{}10000'.format(lang.upper())), os.path.join(args.cldc_input, 'test', lang)]
    for cldc_input in cldc_inputs:
      print(cldc_input)
      for root, dirs, files in os.walk(cldc_input):
        if not files:
          continue
        input_text = []
        for file_name in tqdm(files):
          # find label according to dir path
          label = os.path.basename(root)
          file_name = os.path.join(root, file_name)
          with open(file_name, 'r') as fin:
            texts = fin.readlines()
          input_text.append(texts)
        input_texts[label] = input_text
  
      # split if train
      if 'train' in cldc_input:
        for label, input_text in input_texts.items():
          train_texts[label] = input_text[: len(input_text) - len(input_text) // 10]
          # DETERMINISTIC shuffle, so that each time the instances are the same after shuffling
          random.Random(1234).shuffle(train_texts[label])
          dev_texts[label] = input_text[len(input_text) - len(input_text) // 10:] 
      elif 'test' in cldc_input:
        test_texts = input_texts
    torch.save({'train': train_texts, 'dev': dev_texts, 'test': test_texts}, 'cldc.{}.pth'.format(lang))


def gen_mldoc_pth(args):
  label_conv = {'CCAT': 'C', 'ECAT': 'E', 'MCAT': 'M', 'GCAT': 'G'}
  train_sufs = ['train.1000', 'train.2000', 'train.5000', 'train.10000']
  dt_sufs = ['dev', 'test']
  for lang in args.mldoc_langs:
    text_dict = {}
    for suf in train_sufs + dt_sufs:
      file_name = os.path.join(args.mldoc_input, lang, '{}.{}'.format(lang, suf))
      input_texts = defaultdict(list)
      with open(file_name, 'r') as fin:
        texts = fin.readlines()
      for text in texts:
        text = text.strip()
        label, text = label_conv[text[:text.find('\t')]], text[text.find('\t') + 1:]
        input_texts[label].append(text)
      assert(len(list(input_texts.keys())) == 4)
      text_dict[suf] = input_texts
    # sort train data
    for label in label_conv.values():
      for i in range(1, len(train_sufs)):
        suf = train_sufs[i]
        pre_suf = train_sufs[i - 1]
        print(lang, suf, label)
        ori_list = text_dict[suf][label][:]
        text_dict[suf][label] = text_dict[pre_suf][label] + [text for text in text_dict[suf][label] if text not in text_dict[pre_suf][label]]
        try:
          assert(set(text_dict[suf][label]) == set(ori_list) and len(ori_list) == len(text_dict[suf][label]))
        except:
          dup_list = [item for item, count in Counter(ori_list).items() if count > 1]
          dup_list1 = [item for item, count in Counter(text_dict[suf][label]).items() if count > 1]
          dup_list = list(set(dup_list) - set(dup_list1))
          text_dict[suf][label].extend(dup_list)
          assert(set(text_dict[suf][label]) == set(ori_list) and len(ori_list) == len(text_dict[suf][label]))
    text_dict['train'] = text_dict['train.10000']
    torch.save(text_dict, 'mldoc.{}.pth'.format(lang))


def gen_mlsa_pth(args):
  splits = ['train', 'test', 'unlabeled']
  for lang in args.mlsa_langs:
    for domain in args.mlsa_domains:
      train_texts = {}
      dev_texts = {}
      for split in splits:
        input_texts = defaultdict(list)
        file_name = os.path.join(args.mlsa_input, lang, domain, split + '.review')
        print(file_name)
        root = ET.parse(file_name).getroot()
        for item in root.findall('item'):
          try:
            rating = float(item.find("rating").text)
            text = item.find("text").text.strip()
          except Exception as e:
            print(e)
            continue

          if rating > 3.0:
            label = 'p'
          elif rating < 3.0:
            label = 'n'
          else:
            continue
          input_texts[label].append(text)

        # split if train
        if split == 'train':
          for label, input_text in input_texts.items():
            # DETERMINISTIC shuffle, so that each time the instances are the same after shuffling
            random.Random(1234).shuffle(input_text)
            train_texts[label] = input_text[: len(input_text) - len(input_text) // 10]
            dev_texts[label] = input_text[len(input_text) - len(input_text) // 10:]
        elif split == 'test':
          test_texts = input_texts
        elif split == 'unlabeled':
          ub_texts = input_texts
      torch.save({'train': train_texts, 'dev': dev_texts, 'test': test_texts, 'ub': ub_texts}, 'mlsa.{}.{}.pth'.format(lang, domain))



if __name__ == '__main__':
  args = create_args()
  #preproc_europarl(args)
  gen_dev(args)
  #gen_cldc_pth(args)
  #gen_mldoc_pth(args)
  #gen_mlsa_pth(args)
