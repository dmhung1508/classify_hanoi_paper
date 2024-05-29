
import os
import torch
from utils import *
from arguments import load_args
from data_loader import MyDataLoader
from train import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer,RobertaForSequenceClassification, RobertaConfig, AdamW
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
import config
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def train_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes',
        default= config.bpe_codes,
        required=False,
        type=str,
        help='path to fastBPE BPE'
    )
    args, unknown = parser.parse_known_args()
    bpe = fastBPE(args)


    vocab = Dictionary()
    vocab.add_from_file(config.vocab)

    data_path = config.data_path
    pre_trained = config.model_pretrained
    num_labels = config.num_class
    max_length = config.max_length
    batch_size = config.batch_size
    config1 = RobertaConfig.from_pretrained(
        config.roberta_config, from_tf=False, num_labels = config.num_class, output_hidden_states=False,
    )
    BERT_SA = RobertaForSequenceClassification.from_pretrained(
        config.roberta,
        config=config1
    )
    BERT_SA.cuda()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataloader = MyDataLoader(data_path, max_length, batch_size, bpe, vocab,config.kfold)

    df_train, df_test = dataloader.load_csv(config.data_path)


    N_SPLITS = config.kfold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

    for fold, (_, val_) in enumerate(skf.split(X=df_train, y=df_train.label)):
        df_train.loc[val_, "kfold"] = fold

    train_ne = Trainer(BERT_SA, config, dataloader, df_train, device)

    train_ne.train()

if __name__ == "__main__":
    train_models()