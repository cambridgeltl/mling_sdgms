# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Other parameters
"""

#************************************************************
# Imported Libraries
#************************************************************
import typing
import os

class Params(object):
  def __init__(self, args):
    #------------------------------------------------------------     
    # General 
    #------------------------------------------------------------     
    # Task to perform
    # ['pa', 'xl', 'xl-adv', 'mo', 'cldc', 'semi-cldc', 'aux-semi-cldc', 'xl-semi-cldc', 'aux-xl-semi-cldc', 'mlsa', 'semi-mlsa']
    # pa: parallel data input
    # xl: cross-lingual vae pretraining
    # xl-adv: cross-lingual vae pretraining with language adversarial learning
    # cldc: cross-lingual and monolingual document classification
    # semi-cldc: semi-supervised cldc 
    # aux-semi-cldc: auxiliary semi-supervised 
    # xl-semi-cldc: cross-lingual semi-supervised cldc
    self.task: str = 'cldc'

    # languages
    self.langs: list = args.langs
    # lang -> lang_idx
    self.lang_dict: dict = {lang: i for i, lang in enumerate(self.langs)}
    # number of languages
    self.lang_n = len(self.langs)

    # segmentation
    #[word, spm]
    self.seg_type = 'spm'
    self.seg_model_path: str = args.seg_model_path

    # CUDA
    self.cuda: bool = args.cuda
    
    # write to tensorboard
    self.write_tfboard = False
    # log path
    self.log_path: str = 'ende_de2de_mldoc_4lb_de128_1000_trainenc_swlstm_z_ep5000_es1000'

    # save model path
    self.save_model: str = args.save_model
    # loading model path
    self.load_model: str = args.load_model

    #------------------------------------------------------------
    # Vocab and embedding
    #------------------------------------------------------------
    # vocab path
    self.vocab_path: list = args.vocab_path
    # pretrained embedding path
    self.pretrained_emb_path: str = args.pretrained_emb_path
    # vocab sizes, sorted according to language list
    self.vocab_sizes: list = [10000]
    #self.vocab_sizes: list = [40000, 50000]
    #self.vocab_sizes: list = [50000, 50000]
    # word embedding size
    self.emb_dim = 300
    # word embedding dropout rate
    self.emb_do = 0.2

    #------------------------------------------------------------
    # VAE pretraining
    #------------------------------------------------------------
    # pretraining input file
    self.train_path: str = args.train_path
    self.dev_path: str = args.dev_path

    # corpus length percentile for pretraining
    self.corpus_percentile = 50

    #------------------------------
    # Hyperparameters
    #------------------------------
    # training epochs
    self.ep: int = 500
    # batch size
    self.bs: int = 128
    # test batch size
    self.test_bs: int = 400

    # initial learning rate
    self.init_lr: float = 5e-4

    # early stopping patience
    self.patience = 5
    self.min_delta = 0.25

    # training data scale, how much data to use for training
    self.train_scale = 1
    # validate every n iterations
    self.VAL_EVERY: int = 4000

    #------------------------------
    # NN
    #------------------------------
    # Encoder
    # [lstm, gru]
    self.enc_type = 'lstm'
    # lstm, gru
    self.enc_in_dim = self.emb_dim
    self.enc_hid_dim = 600
    self.enc_num_layers = 2
    self.enc_bidi = True
    self.enc_do = 0.2
    self.x_hid_dim = 2 * self.enc_hid_dim if self.enc_bidi else self.enc_hid_dim

    self.adv_training = False
    # discriminator
    self.xlingdiscriminator_config = [2 * self.enc_hid_dim if self.enc_bidi else self.enc_hid_dim,
                                      1024,
                                      'leakyrelu',
                                      1024,
                                      1
                                     ]
    # inferer input
    self.inf_in_dim = 2 * self.enc_hid_dim if self.enc_bidi else self.enc_hid_dim
    # Z
    self.z_dim = 300
    # Decoder
    #[bow, gru]
    self.dec_type = 'bow'
    # tie embeddings
    self.tie_emb = True
    # lstm
    self.dec_rnn_in_dim = 100
    self.dec_rnn_hid_dim = 50
    self.dec_rnn_num_layers = 1
    self.dec_rnn_do = 0.5

    #------------------------------
    # Tuning KLD
    #------------------------------
    # type of vae losses
    # [detz, 'standard', 'nokld', 'sigmoid', 'fixa']
    # detz: deterministic z
    # standard: vanila vae, optimizer nll + kl
    # nokld: do not optimization kl term, i.e. only optimize nll
    # sigmoid, fixa: nll + alpha * kl
    self.vae_type = 'fixa'
    # sigmoid
    self.sigmoid_x0ep = 4
    self.sigmoid_k = 5e-5
    # beta
    self.beta_C = 10
    self.beta_gamma = 1.0
    # fixed alpha
    self.fixed_alpha: float = 0.1









    #------------------------------------------------------------
    # CLDC, MLDOC
    #------------------------------------------------------------
    # cldc dir
    self.cldc_path: list = args.cldc_path

    # cldc labels
    self.cldc_label2idx = {'C': 0, 'E': 1, 'G': 2, 'M': 3}
    self.cldc_idx2label = {0: 'C', 1: 'E', 2: 'G', 3: 'M'}
    # 23lb
    #self.cldc_label2idx = {'G': 0, 'M': 1}
    #self.cldc_idx2label = {0: 'G', 1: 'M'}
    # 23lb
    self.cldc_label_size = len(self.cldc_label2idx)

    # search and select
    self.ss_file = None #'CEGM_enen.pth'

    # available language data for cldc tasks
    self.cldc_data_langs = args.cldc_data_langs
    self.cldc_data_lang_dict = {lang: i for i, lang in enumerate(self.cldc_data_langs)}
    # train on one data, test on another data
    self.cldc_langs = ['de', 'de']

    # training mode
    #['fixenc', 'trainenc']
    # fixenc: fix both embeddings and encoder
    # trainenc: train everything
    self.cldc_train_mode = 'trainenc'

    self.cldc_classifier_config = ['dropout_0.1',
                                   self.z_dim, #self.x_hid_dim,
                                   #1024,
                                   #'leakyrelu',
                                   #1024,
                                   self.cldc_label_size
                                  ]
    self.aux_hid_dim = 1000
    self.aux_cldc_classifier_config = ['dropout_0.1',
                                   self.aux_hid_dim,
                                   1024,
                                   'leakyrelu',
                                   1024,
                                   self.cldc_label_size
                                   ]
    '''
    self.cldc_classifier_config = [         
                                   self.z_dim, 
                                   1
                                  ]  
    ''' 
    self.cldc_init_lr = 5e-4
    self.cldc_patience = 1000
    # loss threshold to trigger early stopping
    self.cldc_lossth = 2.0
    self.cldc_bs = 16
    self.cldc_test_bs = 1500 # 1500, 2000
    self.cldc_ep = 5000
    self.cldc_warm_up_ep = 5000 # only train classifier
    # source data scale
    self.cldc_train_scale = 0.130
    # target data scale
    self.cldc_xl_train_scale = 0.0
    self.CLDC_VAL_EVERY: int = 8
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

    # en_mldoc_4lb_1000
    # 32: 0.034, 2;
    # 64: 0.066, 4;
    # 128: 0.130, 8;
    # 256: 0.258, 16;
    # 1000: 1.0, 63
    # de_mldoc_4lb_1000
    # 32: 0.0333, 2;
    # 64: 0.066, 4;
    # 128: 0.130, 8;
    # 256: 0.258, 16;
    # 1000: 1.0, 63
    # fr_mldoc_4lb_1000
    # 32: 0.0345, 2;
    # 64: 0.066, 4;
    # 128: 0.130, 8;
    # 256: 0.258, 16;
    # 1000: 1.0, 63


    #------------------------------     
    # SEMI-CLDC 
    #------------------------------     
    # [concat, transconcat, transadd, gmix_transadd]
    self.semicldc_cond_type: str = 'transadd'
    # [uniform, train_prop]
    self.semicldc_yprior_type: str = 'uniform'
    self.semicldc_init_lr = 5e-4
    self.semicldc_classifier_alpha = 0.001 * 1000 / 32 #0.1
    self.semicldc_U_bs = 64

    # linear annealing epochs
    self.semicldc_anneal_warm_up = self.CLDC_VAL_EVERY * 3000

    # cyclic period
    self.cyclic_period = self.CLDC_VAL_EVERY * 500

    #------------------------------
    # XL-SEMI-CLDC
    #------------------------------
    self.src_le = 1.0
    self.src_ue = 0.0
    self.trg_le = 0.0
    self.trg_ue = 1.0
    self.src_cls_alpha = 0.2 * 1064 / 64
    self.trg_cls_alpha = 0.0

    # zero-shot regularization
    self.zs_reg_alpha = 0.0
    # for zero-shot, what parameters should be frozen
    # [encoder, embedding, None]
    self.zs_freeze = None
    # single gru decoder
    self.single_dec = False






    #------------------------------------------------------------     
    # MLSA
    #------------------------------------------------------------     
    # mlsa dir
    self.mlsa_path: list = args.mlsa_path

    # mlsa labels
    self.mlsa_label2idx = {'p': 0, 'n': 1}
    self.mlsa_idx2label = {0: 'p', 1: 'n'}
    self.mlsa_label_size = len(self.mlsa_label2idx)

    # train on one data, test on another data
    self.mlsa_langs = ['en', 'en']

    # training mode
    #['fixenc', 'trainenc']
    # fixenc: fix both embeddings and encoder
    # trainenc: train everything
    self.mlsa_train_mode = 'trainenc'

    self.mlsa_classifier_config = self.cldc_classifier_config[: -1] + [self.mlsa_label_size]
    self.mlsa_init_lr = 5e-4
    self.mlsa_patience = 2000
    # loss threshold to trigger early stopping
    self.mlsa_lossth = 0.005
    self.mlsa_bs = 16
    self.mlsa_test_bs = 1500 # 2048, 1500
    self.mlsa_ep = 2000
    self.mlsa_train_scale = 0.036
    self.MLSA_VAL_EVERY: int = 4
    # en_train: 1.0, 1800, 113


    #------------------------------     
    # MLSA-CLDC 
    #------------------------------     
    # [concat, transconcat, transadd]
    self.semimlsa_cond_type: str = 'concat'
    # [uniform, train_prop]
    self.semimlsa_yprior_type: str = 'uniform'
    self.semimlsa_init_lr = 5e-4
    self.semimlsa_classifier_alpha = 0.113
    self.semimlsa_U_bs = 128
