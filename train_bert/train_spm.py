# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Updated 01/06/2020
"""

#************************************************************
# Imported Libraries
#************************************************************
import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=/yourpath/data/europarl/europarl-v7.de-en.ende.preproc.train'
                               + ' --model_prefix=ende.spm'
                               + ' --vocab_size=10000'
                               + ' --model_type=bpe'
                               + ' --shuffle_input_sentence=true'
                               )
