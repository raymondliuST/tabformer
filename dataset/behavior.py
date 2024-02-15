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
from dataset.vocab import Vocabulary

logger = logging.getLogger(__name__)
log = logger


class BehaviorDataset(Dataset):
    def __init__(self,
                 mlm=False,
                 estids=None,
                 cached=False,
                 root="./data/behavior/",
                 fname="people-model-20240124",
                 fextension="behavior",
                 nrows=None,
                 flatten=True,
                 skip_user=True,
                 user_vocab = None,
                 event_vocab = None):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.estids = estids
        self.skip_user = skip_user

        self.mlm = mlm

        self.flatten = flatten

        self.user_vocab = user_vocab
        self.event_vocab = event_vocab

        self.encoder_fit = {}

        self.behavior_table = None
        self.data = []
        self.labels = []
        self.window_label = []

        self.user_ncols = None
        self.event_ncols = None
        print("Starting encoding data")
        self.encode_data()

        print("Preparing samples")
        self.prepare_samples()


    def __getitem__(self, index):
        user_data = torch.tensor(self.data[index]["input"], dtype=torch.long)
        event_data = torch.tensor(self.data[index]["label"], dtype=torch.long).reshape(-1, self.event_ncols)

        return {"input": user_data, "label": event_data}

    def __len__(self):
        return len(self.data)

    def user_level_data(self):
        # Group data by user estid
        # Total Length will be the number of unique user

        fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        behavior_user_data, behavior_event_data = [], []

        if self.cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))

            behavior_user_data = cached_data["behavior_user_data"]
            behavior_event_data = cached_data["behavior_event_data"]
            user_col_names = cached_data["user_col_names"]
            event_col_names = cached_data["event_col_names"]


        else:
            unique_users = self.behavior_table["estid"].unique()

            event_col_names = ['dayOfWeek', 'dayOfMonth', 'timeOfDay', 'url_category', 'domain_category'] 
            user_col_names = ['os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization'] 

            for user in tqdm.tqdm(unique_users):
                user_all_data = self.behavior_table.loc[self.behavior_table["estid"] == user]

                user_data = list(user_all_data.iloc[0][user_col_names]) # each user can only have one row but events can be multiple
                event_data = [] # flattened event data i.e (,seq_len * ncols)
                for idx, row in user_all_data.iterrows():
                    event_data.extend(row[event_col_names])

                behavior_user_data.append(user_data)
                behavior_event_data.append(event_data)

            assert len(behavior_user_data) == len(behavior_event_data)

            with open(fname, 'wb') as cache_file:
                pickle.dump({"behavior_user_data": behavior_user_data, "behavior_event_data": behavior_event_data, "user_col_names": user_col_names, "event_col_names":event_col_names}, cache_file)

        # convert to str
        return behavior_user_data, behavior_event_data, user_col_names, event_col_names

    def format_data(self, data_lst, column_names, mode = "user"):
        """
            convert each column from local id to global id
            input:
                event_lst: a list of events for one estid
            column_names
            mode: user or event
            output:
        """ 

        if mode == "user":
            vocab = self.user_vocab
        elif mode == "event":
            vocab = self.event_vocab
        
        data_lst = list(divide_chunks(data_lst, len(column_names)))  #

        user_vocab_ids = []
        sep_id = vocab.get_id(vocab.sep_token, special_token=True)
        for data in data_lst:
            vocab_ids = []
            for jdx, field in enumerate(data):
            
                vocab_id = vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            user_vocab_ids.append(vocab_ids)
        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        behavior_user_data, behavior_event_data, user_col_names, event_col_names = self.user_level_data()

        self.user_ncols = len(user_col_names)
        self.event_ncols = len(event_col_names)

        log.info("creating transaction samples with vocab")
        print("preparing user level data...")

        idx = 0
        for user_idx in tqdm.tqdm(range(len(behavior_user_data))):
            
            user_row = behavior_user_data[user_idx]
            event_row = behavior_event_data[user_idx]
            
            user_row_ids = self.format_data(user_row, user_col_names, mode = "user") # global ids 
            event_row_ids = self.format_data(event_row, event_col_names, mode = "event") # global ids 

            # each event is full trace
            ids = event_row_ids
            ids = [idx for ids_lst in ids for idx in ids_lst]

            self.data.append({"input": user_row_ids[0], "label": ids})


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

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        mfit = LabelEncoder()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.behavior_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        # loading event/user label_encoder 
        with open('data/user/preprocessed/user_data_user.encoder_fit.pkl', 'rb') as f:
            user_encoder_fit = pickle.load(f)

        with open('data/event_w_dedupe/preprocessed/event_data_w_dedupe.encoder_fit.pkl', 'rb') as f:
            event_encoder_fit = pickle.load(f)

        self.encoder_fit = {**user_encoder_fit, **event_encoder_fit}

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(self.encoder_fit.keys()):
            col_data = data[col_name]
            
            col_data = self.encoder_fit[col_name].transform(col_data)

            data[col_name] = col_data

        columns_to_select = ['estid','dayOfWeek', 'dayOfMonth', 'timeOfDay', 'url_category', 'domain_category', 
                                'os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization']

        self.behavior_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.behavior_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))
        path.isfile(path.join(dirname, fname))
