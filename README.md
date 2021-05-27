# mling_sdgms
This is the code repo for our EACL 2021 paper [Combining Deep Generative Models and Cross-lingual Petraining for Semi-supervised Document Classification](https://www.aclweb.org/anthology/2021.eacl-main.76.pdf).

The code structure is shown as follows  

```
.
├── data_model
├── nn_model
├── utils
├── train_bert
├── args.py
├── params.py
├── main.py
├── train_xling.py
├── train_semicldc.py
├── train_xlsemicldc.py
├── ...
├── requirements.txt
├── README.md
```

`args.py` and `params.py` contains all the arguments (data path, hyperparameters, etc.) for all experiments.

`main.py` is the main entry of all programs.

`train_xling.py` is used to train our non-parallel cross-lingual VAE (NXVAE).

`train_semicldc.py` is used to perform both supervised and semi-supervised *mono-lingual* document classification.

`train_xlsemicldc.py` is used to perform both supervised and semi-supervised *zero-shot cross-lingual* document classification.

We use `pytorch 1.1.0`, and all library dependencies generated from Conda (there could be MANY useless/redundant dependencies) can be found in `requirements.txt`.

The purpose of each folder is:

`data_model` contains data reading module such as building vocabulary, converting text to its vocabulary id. NOTE that MLDoc (the dataset we use in our experiments) is not publicly available.

`nn_model` contains all the neural model implementations, among which:
- `xlingva.py` is the NXVAE model;
- `semicldc_model.py` is the mono-lingual M1+M2 model;
- `xlsemicldc_model` is the zero-shot M1+M2 model;
- `aux_semicldc_model` is the mono-lingual AUX model;
- `aux_xlsemicldc_model.py` is the zero-shot AUX model;

`utils` contains scripts about IO, preprocessing and the calculation of some common distributions.

`train_bert` contains all the code on pretraining BERT and performing document classification with BERT.
