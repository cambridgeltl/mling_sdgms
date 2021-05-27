# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import os
import matplotlib.pyplot as plt
from scipy.misc import imread
import torch
import numpy as np
import faiss

from args import create_args
from params import Params
from main import get_pretrained_model, get_vocabs, print_info
import train_xling
import train_cldc
from train_cldc import get_batch
from nn_model.xlingva import XlingVA
from nn_model.cldc_model import CLDCModel
from nn_model.semicldc_model import SEMICLDCModel

import pdb

def plot_drplan(df):
  """
  4 settings
  MONO (en)
  MONO (de)
  XLING
  ADVXLING
  """
  beta = beta = df.columns[1:].tolist()
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  ax1.set_xlabel('KL Value')
  ax1.set_ylabel('NLL Value')
  ax2.set_ylabel('Acc')

  # MONO (en)
  mono_kl_en = df[df['beta'] == 'MONO_DEV_KL (en)'].iloc[0][1:].tolist()
  mono_nll_en = df[df['beta'] == 'MONO_DEV_NLL (en)'].iloc[0][1:].tolist()
  mono_cldc_acc_en = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'MONO_MDC_DEV_VAEEMB (en)'].iloc[0][1:].tolist()]
  ax1.plot(mono_kl_en, mono_nll_en, marker= 'o', color = 'r', label = 'MONO EN')
  ax2.plot(mono_kl_en, mono_cldc_acc_en, marker = 'v', color = 'g', label = 'MONO EN')
  for i, txt in enumerate(beta):
    #ax1.annotate(txt, (mono_kl_en[i], mono_nll_en[i]))
    ax2.annotate(mono_cldc_acc_en[i], (mono_kl_en[i], mono_cldc_acc_en[i]))

  # MONO (de)
  mono_kl_de = df[df['beta'] == 'MONO_DEV_KL (de)'].iloc[0][1:].tolist()
  mono_nll_de = df[df['beta'] == 'MONO_DEV_NLL (de)'].iloc[0][1:].tolist()
  mono_cldc_acc_de = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'MONO_MDC_DEV_VAEEMB (de)'].iloc[0][1:].tolist()]
  ax1.plot(mono_kl_de, mono_nll_de, marker= 'o', color = 'tomato', label = 'MONO DE')
  ax2.plot(mono_kl_de, mono_cldc_acc_de, marker = 'v', color = 'lightgreen', label = 'MONO DE')
  for i, txt in enumerate(beta):
    #ax1.annotate(txt, (mono_kl_de[i], mono_nll_de[i]))
    ax2.annotate(mono_cldc_acc_de[i], (mono_kl_de[i], mono_cldc_acc_de[i]))

  # XLING
  xling_kl = df[df['beta'] == 'XLING_DEV_KL'].iloc[0][1:].tolist()
  xling_nll = df[df['beta'] == 'XLING_DEV_NLL'].iloc[0][1:].tolist()
  xling_cldc_acc_en = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'XLING_MDC_DEV_VAEEMB (en)'].iloc[0][1:].tolist()]
  xling_cldc_acc_de = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'XLING_MDC_DEV_VAEEMB (de)'].iloc[0][1:].tolist()]
  ax1.plot(xling_kl, xling_nll, marker= 'o', color = 'fuchsia', label = 'XLING')
  ax2.plot(xling_kl, xling_cldc_acc_en, marker = 'v', color = 'darkblue', label = 'XLING EN')
  ax2.plot(xling_kl, xling_cldc_acc_de, marker = 'v', color = 'cornflowerblue', label = 'XLING DE')
  for i, txt in enumerate(beta):
    #ax1.annotate(txt, (xling_kl[i], xling_nll[i]))
    ax2.annotate(xling_cldc_acc_en[i], (xling_kl[i], xling_cldc_acc_en[i]))
    ax2.annotate(xling_cldc_acc_de[i], (xling_kl[i], xling_cldc_acc_de[i]))

  # ADVXLING
  advxling_kl = df[df['beta'] == 'ADVXLING_DEV_KL'].iloc[0][1:].tolist()
  advxling_nll = df[df['beta'] == 'ADVXLING_DEV_NLL'].iloc[0][1:].tolist()
  advxling_cldc_acc_en = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'ADVXLING_MDC_DEV_VAEEMB (en)'].iloc[0][1:].tolist()]
  advxling_cldc_acc_de = [float("{:.3f}".format(i)) for i in df[df['beta'] == 'ADVXLING_MDC_DEV_VAEEMB (de)'].iloc[0][1:].tolist()]
  ax1.plot(advxling_kl, advxling_nll, marker= 'o', color = 'violet', label = 'ADVXLING')
  ax2.plot(advxling_kl, advxling_cldc_acc_en, marker = 'v', color = 'gold', label = 'ADVXLING EN')
  ax2.plot(advxling_kl, advxling_cldc_acc_de, marker = 'v', color = 'yellow', label = 'ADVXLING DE')
  for i, txt in enumerate(beta):
    #ax1.annotate(txt, (advxling_kl[i], advxling_nll[i]))
    ax2.annotate(advxling_cldc_acc_en[i], (advxling_kl[i], advxling_cldc_acc_en[i]))
    ax2.annotate(advxling_cldc_acc_de[i], (advxling_kl[i], advxling_cldc_acc_de[i]))

  ax1.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=True)
  ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fancybox=True, shadow=True)

  plt.show()
  fig.savefig('{}.pdf'.format('pretrain_mdc'))


def gen_z_pretrain(params, datas, xlingva):
  xlingva.eval()
  bs = 5000
  for lang, lang_idx in params.lang_dict.items():
    with open('./z.{}.out'.format(lang), 'w') as fout:
      data = datas[lang_idx]
      n_batch = data.dev_size // bs if data.dev_size % bs == 0 else data.dev_size // bs + 1
      data_idxs = list(range(data.dev_size))
      fout.write('{} {}\n'.format(data.dev_size, params.z_dim))

      for k in range(n_batch):
        test_idxs = data_idxs[k * bs: (k + 1) * bs]
        # get padded & sorted batch idxs and 
        with torch.no_grad():
          padded_batch, batch_lens = train_xling.get_batch(test_idxs, data, data.dev_idxs, data.dev_lens, params.cuda)      
          mu, logvar = xlingva.get_gaus(lang, padded_batch, batch_lens)
          batch_text = data.idx2text(padded_batch.cpu().tolist(), idx_lens = batch_lens.tolist())
          assert(len(batch_text) == mu.shape[0])
          mu = mu.cpu().tolist()
          z_embs = list(zip(batch_text, mu))
          z_embs = ['{} {}'.format(w[0], ' '.join(list(map(lambda x: str(x), w[1])))) for w in z_embs]
          fout.write('{}\n'.format('\n'.join(z_embs)))


def gen_z_cldc(params, datas, m):
  m.eval()
  bs = 5000

  for lang, lang_idx in params.lang_dict.items():
    with open('./z.cldc.{}.out'.format(lang), 'w') as fout:
      data = datas[lang_idx]
      n_batch = data.dev_size // bs if data.dev_size % bs == 0 else data.dev_size // bs + 1
      data_idxs = [list(range(len(dev_idx))) for dev_idx in data.dev_idxs]
      fout.write('{} {}\n'.format(data.dev_size, params.z_dim))

      for j in range(n_batch):
        test_idxs = []
        for k, data_idx in enumerate(data_idxs):
          if j < n_batch - 1:
            test_idxs.append(data_idx[j * int(bs * data.dev_prop[k]): (j + 1) * int(bs * data.dev_prop[k])])
          elif j == n_batch - 1:
            test_idxs.append(data_idx[j * int(bs * data.dev_prop[k]):])

        with torch.no_grad():
          batch_in, batch_lens, batch_lb = train_cldc.get_batch(params, test_idxs, data.dev_idxs, data.dev_lens)
          mu, logvar = m.get_gaus(lang, batch_in, batch_lens)
          batch_text = data.idx2text(batch_in.cpu().tolist(), batch_lb.tolist(), idx_lens = batch_lens.tolist())
          assert(len(batch_text) == mu.shape[0])
          mu = mu.cpu().tolist()
          z_embs = list(zip(batch_text, mu))
          z_embs = ['{} {}'.format(w[0], ' '.join(list(map(lambda x: str(x), w[1])))) for w in z_embs]
          fout.write('{}\n'.format('\n'.join(z_embs)))


def show_lasth(img_dir):
  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()
  plt.ion()
  for root, dirs, files in os.walk(img_dir):
    files.sort(key=lambda x: os.stat(os.path.join(root, x)).st_mtime)
    for i, file_name in enumerate(files):
      img = imread(os.path.join(root, file_name))
      plt.imshow(img)
      plt.axis('off')
      plt.tight_layout()
      plt.title('Number {}'.format(i))
      plt.pause(0.1)
      plt.clf()


def get_mu1(params, m, input_idxs, input_lens, input_size, lang):
  m.eval()
  n_batch = input_size // params.cldc_test_bs if input_size % params.cldc_test_bs == 0 else input_size // params.cldc_test_bs + 1
  data_idxs = [list(range(len(input_idx))) for input_idx in input_idxs]

  mu1s = []
  test_idxs = []
  for i, data_idx in enumerate(data_idxs):
    test_idxs = [[]] * i + [data_idx] + [[]] * (len(data_idxs) - 1 - i)
    with torch.no_grad():
      batch_in, batch_lens, batch_lb = get_batch(params, test_idxs, input_idxs, input_lens)
      if params.task == 'cldc':
        # embedding
        input_word_embs = m.embeddings(lang, batch_in)
        # encoding
        hid = m.encoder(input_word_embs, batch_lens)
        # infering
        mu1, logvar1 = m.inferer(hid)
      elif 'semi' in params.task:
        mu1, logvar1, z1 = m.get_z1(lang, batch_in, batch_lens)
      mu1s.append(mu1)
  return mu1s


def search_and_select(dev_xs, train_xs):
  y_names = ['C', 'E', 'G', 'M']
  y_idxs = [0, 1, 2, 3]
  idx_map = {}
  for idx, label in zip(y_idxs, y_names):
    xq = torch.mean(dev_xs[idx], dim = 0).unsqueeze(0).cpu().numpy()
    xb = train_xs[idx].cpu().numpy()
    d = xb.shape[-1]
    k = xb.shape[0]
    # faiss search
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    D, I = index.search(xq, k)
    idx_map[idx] = I
  torch.save(idx_map, '{}.pth'.format(''.join(y_names)))


if __name__ == '__main__':
# [vis_z, vis_lasth, sel_exp]
  vis_type = 'sel_exp'

  arguments = create_args()
  params = Params(arguments)

  # load pretrained model
  vocab_dict, model_dict = get_pretrained_model(params)
  # get vocabs for different languages
  vocabs = get_vocabs(params, vocab_dict)
  # general info
  print_info(params, vocabs)

  datas = train_cldc.get_data(params, vocabs)
  assert (len(datas) == 2)

  train_lang, train_data = train_cldc.get_lang_data(params, datas, training = True)
  test_lang, test_data = train_cldc.get_lang_data(params, datas)

  # use pretraining data & model
  if vis_type == 'vis_z':
    gen_z_cldc(params, datas, m)
    # gen z for pretrained corpus
    '''
    xlingva = XlingVA(params, datas, model_dict = model_dict)
    gen_z_pretrain(params, datas, xlingva)
    '''
    # gen z for cldc corpus
    m = CLDCModel(params, datas, model_dict=model_dict)
  elif vis_type == 'vis_lasth':
    if params.task == 'cldc':
      m = CLDCModel(params, datas, params.cldc_classifier_config)
    elif params.task == 'semi-cldc':
      m = SEMICLDCModel(params, datas)

    # get the current module parameter dicts
    cur_model_dict = m.state_dict()
    # 1. filter out unnecessary keys
    # new code model
    filtered_model_dict = {k: model_dict[k] for k in cur_model_dict}
    assert (set(filtered_model_dict.keys()) == set(cur_model_dict.keys()))
    # 2. overwrite entries in the existing state dict
    cur_model_dict.update(filtered_model_dict)
    # 3. load the new state dict
    m.load_state_dict(cur_model_dict)

    train_cldc.gen_task_info(params, m, datas)

    '''
    train_cldc.test(params, m, test_data.dev_idxs, test_data.dev_lens, test_data.dev_size,
                    test_data.dev_prop, test_lang, vis = True)
    '''
    train_cldc.test(params, m, test_data.test_idxs, test_data.test_lens, test_data.test_size,
                    test_data.test_prop, test_lang, vis = True)
    train_cldc.tsne2d(params, m)

    '''
    img_dir = sys.argv[1]
    show_lasth(img_dir)
    '''
  elif vis_type == 'sel_exp':
    train_size = 16
    if params.task == 'cldc':
      m = CLDCModel(params, datas, params.cldc_classifier_config, model_dict = model_dict)
    elif params.task == 'semi-cldc':
      m = SEMICLDCModel(params, datas, model_dict = model_dict)

    #y_names, y_idxs = params.cldc_label2idx.keys(), params.cldc_label2idx.values()
    train_cldc.gen_task_info(params, m, datas)

    dev_xs = get_mu1(params, m, test_data.dev_idxs, test_data.dev_lens, test_data.dev_size, test_lang)
    train_xs = get_mu1(params, m, train_data.train_idxs, train_data.train_lens, train_data.train_size, train_lang)

    search_and_select(dev_xs, train_xs)


  '''
  in_path = sys.argv[1]
  df = pd.read_csv(in_path, sep = '\t')
  plot_drplan(df)
  '''


