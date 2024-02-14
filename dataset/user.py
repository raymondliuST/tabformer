import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from dataset.vocab import Vocabulary, load_vocab

logger = logging.getLogger(__name__)
log = logger


class UserDataset(Dataset):
    def __init__(self,
                 mlm=True,
                 estids=None,
                 cached=False,
                 root="./data/user/",
                 fname="user_data",
                 vocab_dir="vocab",
                 fextension="user",
                 nrows=None,
                 flatten=True,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=True):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.estids = estids
        self.return_labels = return_labels
        self.skip_user = skip_user

        self.mlm = mlm

        self.flatten = flatten

        self.vocab = Vocabulary(adap_thres)
        self.encoder_fit = {}

        self.user_table = None
        self.data = []
        self.labels = []
        self.window_label = []

        self.ncols = None
        print("Starting encoding data")
        self.encode_data()
        print("Initiating vocab")
        self.init_save_vocab(vocab_dir)
        print("Preparing samples")
        self.prepare_samples()
        # print("Saving vocab")
        # self.save_vocab(vocab_dir)

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))

        return return_data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    def user_level_data(self):
        # Group trans data by user estid
        # Total Length will be the number of unique user
        # For each user 

        fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []

        if self.cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]

        else:
            unique_users = self.user_table["estid"].unique()
            columns_names = list(self.user_table.columns)

            for idx, row in self.user_table.iterrows():
                row = list(row)

                # assumption that user is first field
                skip_idx = 1 if self.skip_user else 0

                trans_data.append(row[skip_idx:])
                trans_labels.append(row[-1]) # not used

            if self.skip_user:
                columns_names.remove("estid")

            with open(fname, 'wb') as cache_file:
                pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

        # convert to str
        return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        # Convert from local id to global id. 
        # Add seperation token after each trans
        
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 1))  # 2 to ignore isFraud and SPECIAL
        
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)
        
        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):

                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            # if self.mlm and self.flatten:  # only add [SEP] for BERT + flatten scenario
            #     vocab_ids.append(sep_id)
 
            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        trans_data, trans_labels, columns_names = self.user_level_data()

        log.info("creating transaction samples with vocab")
        print("preparing user level data...")

        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]

            user_row_ids = self.format_trans(user_row, columns_names)
            user_labels = trans_labels[user_idx]

            bos_token = self.vocab.get_id(self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(self.vocab.eos_token, special_token=True)  # will be used for GPT2

            self.data.append(user_row_ids[0])


        '''
            ncols = total fields - 1 (special tokens)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)

        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.estids:
            log.info(f'Filtering data by user ids list: {self.estids}...')
            self.estids = map(int, self.estids)
            data = data[data['estid'].isin(self.estids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_save_vocab(self, vocab_dir):

        file_name = path.join(vocab_dir, f'vocab{self.fextension}.json')

        if self.cached:
            self.vocab = load_vocab(file_name)
            return

        column_names = list(self.user_table.columns)
        if self.skip_user:
            column_names.remove("estid")

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.user_table[column].value_counts(sort=True).to_dict()  # returns sorted
            class_weight_col = [sum(unique_values.values())/count for cls, count in unique_values.items()]

            self.vocab.column_weights[column] = class_weight_col # setting up column weights

            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.user_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        sub_columns = ['os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization']

        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        columns_to_select = ['estid', 'os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization']

        self.user_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.user_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))
        path.isfile(path.join(dirname, fname))