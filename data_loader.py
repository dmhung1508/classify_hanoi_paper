import config
import torch
from utils import *
from torch.utils.data import (DataLoader, TensorDataset)
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

MAX_LEN = config.max_length
fold = config.kfold
class MyDataLoader():
    def __init__(self, dataset_path, max_length, batch_size, bpe, vocab,test_sample = False,):
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.test_sample = test_sample
        self.bpe = bpe
        self.vocab = vocab

    def load_csv(self, dataset_path):
        df = pd.read_csv(dataset_path)
        labels = df.label.values.tolist()

        label_0 = [i for i in labels if int(i) == 0]
        label_1 = [i for i in labels if int(i) == 1]
        label_2 = [i for i in labels if int(i) == 2]

        print('has {} label 0 in {} total label'.format(len(label_0), len(labels)))
        print('has {} label 1 in {} total label'.format(len(label_1), len(labels)))
        print('has {} label 2 in {} total label'.format(len(label_2), len(labels)))
        df = preprocess(df)
        df = df[['text', 'label']]
        df.loc[:, 'text'] = df['text'].str.lower()
        df = df.reset_index(drop=True)
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=55)
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()

        return df_train,df_test

    def prepare_loaders(self,fold,df_train):
        df_train1 = df_train[df_train.kfold != fold].reset_index(drop=True)
        df_valid1 = df_train[df_train.kfold == fold].reset_index(drop=True)
        train_text = []
        train_labels = []
        for i in tqdm(df_train1.index):
            train_text.append(tokenlizes(df_train1.iloc[i]['text']))
            train_labels.append(df_train1.iloc[i]['label'])

        val_text = []
        val_labels = []
        for i in tqdm(df_valid1.index):
            val_text.append(tokenlizes(df_valid1.iloc[i]['text']))
            val_labels.append(df_valid1.iloc[i]['label'])
        train_ids = []
        for sent in train_text:
            subwords = '<s> ' + self.bpe.encode(sent) + ' </s>'
            encoded_sent = self.vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
            train_ids.append(encoded_sent)

        val_ids = []
        for sent in val_text:
            subwords = '<s> ' + self.bpe.encode(sent) + ' </s>'
            encoded_sent = self.vocab.encode_line(subwords, append_eos=True,
                                             add_if_not_exist=False).long().tolist()  # Ánh xạ subword vào vocab để trích xuất tensor tương ứng
            val_ids.append(encoded_sent)
        train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                  padding="post")  # Pad để đưa về cùng size input nếu nhỏ hơn max length thì
        #  sẽ pad cho đủ length, còn dài hơn thì sẽ cắt tại max length
        val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        # Mask để mô hình chú ý vào phần câu, bỏ qua phần padding
        train_masks = []
        for sent_train in train_ids:
            mask = [int(token_id > 0) for token_id in sent_train]
            train_masks.append(mask)

        val_masks = []
        for sent_val in val_ids:
            mask = [int(token_id > 0) for token_id in sent_val]
            val_masks.append(mask)

        train_inputs = torch.tensor(train_ids)
        val_inputs = torch.tensor(val_ids)

        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)

        train_masks = torch.tensor(train_masks)
        val_masks = torch.tensor(val_masks)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler =SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)

        return train_dataloader, val_dataloader
