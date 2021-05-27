import sys
sys.path.append('/yourpath/')

import argparse
import torch
from torch import optim
import random
from random import shuffle
from transformers import BertForSequenceClassification, BertConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import copy

import tokenization
from bert_cldc_data_reader import BERTCLDCDataReader
from utils.early_stopping import EarlyStopping
from train_cldc import get_batch, get_classification_report, get_cm
from utils.ios import out_cldc


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

import pdb

def create_args():
  parser = argparse.ArgumentParser(description = 'BERT Classification')

  # tokenizer
  parser.add_argument("--vocab_file", type = str,
                      default = '/yourpath/data/bert/ende.spm.vocab.list',
                      #default = '/yourpath/data/bert/ende.word.list',
                      help = "vocab file for tokenizer")
  parser.add_argument("--do_lower_case", action='store_true', default = True,
                      help = "lower case")
  parser.add_argument("--piece", type = str,
                      default = 'sentence',
                      #default = 'word_model',
                      help = "sentence piece or word piece")
  parser.add_argument("--piece_model", type = str,
                      default = '/yourpath/data/bert/ende.spm.model',
                      #default = None,
                      help = "piece model")
  # Bert
  parser.add_argument("--config_file", type = str,
                      default = '/yourpath/data/bert/bert_classification_config_swlstm_simpar.json',
                      #default = '/yourpath/data/bert/bert_classification_config_word.json',
                      help = "bert config file")
  parser.add_argument("--pretrained_file", type = str,
                      default = '/yourpath/data/bert/bert.pretrained.swlstm.simpar.pth',
                      #default = '/yourpath/data/bert/bert.pretrained.word.pth',
                      help = "pretrained file")
  # data
  parser.add_argument("--cldc_path", nargs = '+',
                      #default = ['/yourpath/data/cldc/cldc.en.pth',
                                 #'/yourpath/data/cldc/cldc.de.pth'],
                      default = ['/yourpath/data/mldoc/en/mldoc.en.pth',
                                 '/yourpath/data/mldoc/de/mldoc.de.pth'],
                      help = "CLDC file path, sorted according to language list")
  parser.add_argument("--cldc_lang", nargs = '+',
                      default = ['en', 'en'],
                      help = "CLDC language for the task")
  parser.add_argument("--scale", type = float,
                      default = 0.034,
                      help = "scale")
  parser.add_argument("--VAL_EVERY", type = int,
                      default = 2,
                      help = "evaluate after how many iterations")
  # ende_en_cldc_4lb_8961
  # 64: 0.00735, 4
  # 128: 0.0145, 8
  # 256: 0.0288, 16
  # 512: 0.0574, 32
  # 1024: 0.11458, 64
  # 8961: 1.0, 561
  # ende_de_cldc_4lb_9002
  # 64: 0.00727, 4
  # 128: 0.01453, 8
  # 256: 0.0287, 16
  # 512: 0.0571, 32
  # 1024: 0.11394, 64
  # 9002: 1.0, 563

  # cuda
  parser.add_argument("--cuda", action='store_true', default=True,
                      help="enable cuda")

  args = parser.parse_args()
  args.cuda = args.cuda if torch.cuda.is_available() else False

  args.init_lr = 2e-5
  args.patience = 1000
  # loss threshold to trigger early stopping
  args.lossth = 2.0
  args.bs = 16
  args.test_bs = 1000
  args.ep = 5000

  args.cldc_label2idx = {'C': 0, 'E': 1, 'G': 2, 'M': 3}
  args.cldc_idx2label = {0: 'C', 1: 'E', 2: 'G', 3: 'M'}

  # [None, 'embedding', 'encoder']
  args.zs_freeze = None

  # for semi-supervised learning
  args.semi_warm_up = 5000
  args.self_train = False

  args.ss_file = None
  #args.log_path = 'ende_de2de_cldc_de64_9002_trainenc_BERTswlstmsimpar_ep1500_es500'
  args.log_path = 'ende_en2en_mldoc_4lb_en32_1000_trainenc_BERTsw_ep5000_es1000'
  return args


def main(params, m, data):
  # early stopping
  es = EarlyStopping(mode='max', patience=params.patience)
  # set optimizer
  optimizer = get_optimizer(params, m)

  n_batch = data.train_size // params.bs if data.train_size % params.bs == 0 else data.train_size // params.bs + 1
  # per category
  data_idxs = [list(range(len(train_idx))) for train_idx in data.train_idxs]

  # number of iterations
  cur_it = 0
  # best xx
  bdev = 0
  btest = 0
  # current xx
  cdev = 0
  ctest = 0
  dev_class_acc = {}
  test_class_acc = {}
  dev_cm = None
  test_cm = None
  # early stopping warm up flag, start es after some iters
  es_flag = False

  for i in range(params.ep):
    # self-training
    if params.self_train or i >= params.semi_warm_up:
      params.self_train = True
      first_update = (i == params.semi_warm_up)
      # only for zero-shot
      if first_update:
        es.num_bad_epochs = 0
        es.best = 0
        bdev = 0
        btest = 0
      data = self_train_merge_data(params, m, es, data, first = first_update)
      n_batch = data.self_train_size // params.bs if data.self_train_size % params.bs == 0 else data.self_train_size // params.bs + 1
      # per category
      data_idxs = [list(range(len(train_idx))) for train_idx in data.self_train_idxs]

    for data_idx in data_idxs:
      shuffle(data_idx)
    for j in range(n_batch):
      train_idxs = []
      for k, data_idx in enumerate(data_idxs):
        if params.self_train:
          train_prop = data.self_train_prop
        else:
          train_prop = data.train_prop
        if j < n_batch - 1:
          train_idxs.append(data_idx[int(j * params.bs * train_prop[k]): int( (j + 1) * params.bs * train_prop[k])])
        elif j == n_batch - 1:
          train_idxs.append(data_idx[int(j * params.bs * train_prop[k]):])

      if params.self_train:
        batch_train, _, batch_train_lb = get_batch(params, train_idxs,
                                                   data.self_train_idxs,
                                                   data.self_train_lens)
      else:
        batch_train, _, batch_train_lb = get_batch(params, train_idxs,
                                                   data.train_idxs,
                                                   data.train_lens)
      optimizer.zero_grad()
      m.train()

      loss_batch, logits = m(batch_train, labels = batch_train_lb)
      batch_pred = torch.argmax(logits, dim = 1)

      batch_acc, batch_acc_cls = get_classification_report(params,
                                                           batch_train_lb.data.cpu().numpy(),
                                                           batch_pred.data.cpu().numpy())

      if loss_batch < params.lossth:
        es_flag = True

      loss_batch.backward()
      out_cldc(i, j, n_batch, loss_batch, batch_acc, batch_acc_cls, bdev, btest, cdev, ctest,
                 es.num_bad_epochs)

      optimizer.step()
      cur_it += 1

    sys.stdout.write('\n')
    sys.stdout.flush()
    # validation
    cdev, dev_class_acc, dev_cm = test(params, m, data.dev_idxs, data.dev_lens,
                                       data.dev_size, data.dev_prop, cm = True)
    ctest, test_class_acc, test_cm = test(params, m, data.test_idxs, data.test_lens,
                                          data.test_size, data.test_prop, cm = True)
    print(dev_cm)
    print(test_cm)
    if es.step(cdev):
      print('\nEarly Stoped.')
      return
    elif es.is_better(cdev, bdev):
      bdev = cdev
      btest = ctest
    # reset bad epochs
    if not es_flag:
      es.num_bad_epochs = 0


def test(params, m, input_idxs, input_lens, input_size, input_prop, cm=False):
  m.eval()
  n_batch = input_size // params.test_bs if input_size % params.test_bs == 0 else input_size // params.test_bs + 1
  data_idxs = [list(range(len(input_idx))) for input_idx in input_idxs]
  acc = .0
  labels = list(params.cldc_label2idx.keys())

  preds = []
  sorted_y = []

  for j in range(n_batch):
    test_idxs = []
    for k, data_idx in enumerate(data_idxs):
      if j < n_batch - 1:
        test_idxs.append(data_idx[int(j * params.test_bs * input_prop[k]): int(
          (j + 1) * params.test_bs * input_prop[k])])
      elif j == n_batch - 1:
        test_idxs.append(data_idx[int(j * params.test_bs * input_prop[k]):])

    with torch.no_grad():
      batch_test, _, batch_test_lb = get_batch(params, test_idxs, input_idxs, input_lens)

      _, logits = m(batch_test, labels = batch_test_lb)
      batch_pred = torch.argmax(logits, dim = 1)


    acc += accuracy_score(batch_test_lb.data.cpu().numpy(), batch_pred.data.cpu().numpy(),
                          normalize=False)
    preds += batch_pred.data.cpu().tolist()
    sorted_y += batch_test_lb.data.cpu().tolist()

  acc /= input_size
  _, recall, _, support = precision_recall_fscore_support(np.array(sorted_y), np.array(preds))
  recall = dict(zip(labels, recall))

  if cm is True:
    cm = get_cm(np.array(sorted_y), np.array(preds))
    return acc, recall, cm
  else:
    return acc, recall


def get_optimizer(params, m):
  ps = {p[0]: p[1] for p in m.named_parameters()}
  if params.zs_freeze == None:
    ps = [p[1] for p in ps.items()]
  elif params.zs_freeze == 'embedding':
    ps = [p[1] for p in ps.items()
          if 'word_embeddings' not in p[0] and
             'position_embeddings' not in p[0] and
             'token_type_embeddings' not in p[0]]
  elif params.zs_freeze == 'encoder':
    ps = [p[1] for p in ps.items() if 'classifier' in p[0]]
  #ps = list(filter(lambda p: p.requires_grad, m.parameters()))
  ps = list(filter(lambda p: p.requires_grad, ps))
  print('Model parameter: {}'.format(sum(p.numel() for p in ps)))
  optimizer = optim.Adam(ps, lr = params.init_lr)
  return optimizer


def self_train_merge_data(params, m, es, data, first = False):
  if first or es.num_bad_epochs == 0:
    print('SELF-TRAINING: update unlabeled data ...')
    # do self-training
    m.eval()
    # for sure we will only have one batch
    with torch.no_grad():
      # sort in the descending order
      sorted_len_idxs = np.argsort(-data.rest_train_lens)
      sorted_batch_x_lens = data.rest_train_lens[sorted_len_idxs]
      sorted_batch_x = data.rest_train_idxs[sorted_len_idxs]
      sorted_batch_x = torch.LongTensor(sorted_batch_x)
      if params.cuda:
        sorted_batch_x = sorted_batch_x.cuda()

      outputs = m(sorted_batch_x)
      logits = outputs[0]
      batch_pred = torch.argmax(logits, dim = 1)
      sorted_batch_x = sorted_batch_x.cpu()

    data.self_train_idxs = copy.deepcopy(data.train_idxs)
    data.self_train_lens = copy.deepcopy(data.train_lens)
    data.self_train_size = data.train_size + data.rest_train_size
    for i, label in enumerate(batch_pred):
      label = int(label)
      data.self_train_idxs[label] = torch.cat((data.self_train_idxs[label], sorted_batch_x[i].unsqueeze(0)))
      data.self_train_lens[label] = np.concatenate([data.self_train_lens[label], np.array([sorted_batch_x_lens[i]])])
    data.self_train_prop = np.array([len(train_idx) for train_idx in data.self_train_idxs])
    data.self_train_prop = (data.self_train_prop / sum(data.self_train_prop)) if sum(data.self_train_prop) != 0 else [0] * len(data.self_train_prop)
  return data


if __name__ == '__main__':
  args = create_args()

  # load tokenizer
  tokenizer = tokenization.FullTokenizer(
    vocab_file = args.vocab_file, do_lower_case = args.do_lower_case,
    piece = args.piece, piece_model = args.piece_model)

  # load bert model
  config = BertConfig.from_json_file(args.config_file)
  model = BertForSequenceClassification(config)
  model_state_dict = model.state_dict()
  print('Model parameter: {}'.format(sum(p.numel() for k, p in model_state_dict.items())))
  pre_state_dict = torch.load(args.pretrained_file)
  pre_state_dict = {k: v for k, v in pre_state_dict.items() if k in model_state_dict}
  model_state_dict.update(pre_state_dict)
  model.load_state_dict(model_state_dict)
  if args.cuda:
    model.cuda()

  # load data
  data = BERTCLDCDataReader(args, tokenizer)

  # general info for cldc
  cldc_log = ('CLDC lang: {}\n'.format(', '.join(args.cldc_lang)) +
              'Label size: {}\n'.format(data.label_size) +
              'Labels: [{}]\n'.format(' '.join([lb for idx, lb in data.idx2label.items()])) +
              'Train percentage: {}\n'.format(args.scale) +
              'Val every: {}\n'.format(args.VAL_EVERY) +
              'Train size: {} [{}]\n'.format(data.train_size, ' '.join([str(len(idx)) for idx in data.train_idxs])) +
              'Dev size: {} [{}]\n'.format(data.dev_size, ' '.join([str(len(idx)) for idx in data.dev_idxs])) +
              'Test size: {} [{}]\n'.format(data.test_size, ' '.join([str(len(idx)) for idx in data.test_idxs])) +
              'Sentence lenper: {}\n'.format(data.max_text_len) +
              'Early stopping: {} {}\n'.format(args.patience, args.lossth) +
              'Batch size: {} {}'.format(args.bs, args.test_bs)
              )
  print(cldc_log)

  # do classification
  main(args, model, data)
