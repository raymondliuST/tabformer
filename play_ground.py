from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from torch.utils.data import DataLoader
from dataset.event import EventDataset
from dataset.user import UserDataset


from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling

import pandas as pd

import torch


# dataset = UserDataset(mlm=True,
#                  estids=None,
#                  cached=False,
#                  root="./data/user/",
#                  fname="user_data",
#                  vocab_dir="vocab",
#                  fextension="user",
#                  nrows=None,
#                  flatten=True,
#                  adap_thres=10 ** 8,
#                  return_labels=False,
#                  skip_user=True)

dataset = EventDataset(mlm=True,
                 estids=None,
                 seq_len=6,
                 cached=True,
                 root="./data/event_w_dedupe/",
                 fname="event_data",
                 vocab_dir="vocab",
                 fextension="event_w_dedupe",
                 nrows=None,
                 flatten=False,
                 stride=2,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=True)

vocab = dataset.vocab

custom_special_tokens = vocab.get_special_tokens()

tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=True,
                               flatten=False,
                               ncols=dataset.ncols,
                               field_hidden_size=80
                               )

collactor_cls = "TransDataCollatorForLanguageModeling"
data_collator = eval(collactor_cls)(
        tokenizer=tab_net.tokenizer, mlm=True, mlm_probability=0.15
    )
model = tab_net.model
batch_size=16
loader = DataLoader(dataset, batch_size=batch_size, collate_fn = data_collator, shuffle=True)

for batch in loader:
    outputs = model(**batch)
    import pdb
    pdb.set_trace()
    


