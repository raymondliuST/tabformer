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
from dataset.behavior import BehaviorDataset

from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig
from models.tabformer_bart import TabFormerBart, TabFormerBartConfig

from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import EventDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling, BehaviorDataCollatorForLanguageModeling

import pandas as pd
from transformers import AdamW

import torch


user_dataset = UserDataset(mlm=True,
                 estids=None,
                 cached=True,
                 root="./data/user/",
                 fname="user_data",
                 vocab_dir="vocab",
                 fextension="user",
                 nrows=None,
                 flatten=True,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=True)

event_dataset = EventDataset(mlm=True,
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


user_vocab = user_dataset.vocab
event_vocab = event_dataset.vocab
custom_special_tokens = user_vocab.get_special_tokens()

pad_token = event_vocab.get_id(event_vocab.pad_token, special_token=True)
bos_token = event_vocab.get_id(event_vocab.bos_token, special_token=True)  # will be used for GPT2
eos_token = event_vocab.get_id(event_vocab.eos_token, special_token=True)

beh_dataset = BehaviorDataset(
                 mlm=False,
                 estids=None,
                 seq_len=None,
                 cached=True,
                 root="./data/behavior/",
                 fname="people-model-20240124",
                 vocab_dir="vocab",
                 fextension="behavior",
                 nrows=None,
                 flatten=True,
                 stride=2,
                 adap_thres=10 ** 8,
                 skip_user=True,
                 user_vocab=user_vocab,
                 event_vocab=event_vocab)

config = TabFormerBartConfig(
        encoder_vocab_size=len(user_vocab),
        decoder_vocab_size=len(event_vocab),
        encoder_ncols=beh_dataset.user_ncols,
        decoder_ncols=beh_dataset.event_ncols,
        d_model=60,
        dropout = 0.1,
        field_hidden_size=64,
        encoder_layers=1,
        encoder_ffn_dim=128,
        encoder_attention_heads=user_dataset.ncols,
        decoder_layers=1,
        decoder_ffn_dim=128,
        decoder_attention_heads=event_dataset.ncols,
        pad_token_id=pad_token,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
        decoder_start_token_id=eos_token
        )

tab_net = TabFormerBart(config, user_vocab, event_vocab)


from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel
)
event_tokenizer = BertTokenizer(event_vocab.filename, do_lower_case=False, **custom_special_tokens)
user_tokenizer = BertTokenizer(user_vocab.filename, do_lower_case=False, **custom_special_tokens)
collactor_cls = "BehaviorDataCollatorForLanguageModeling"
data_collator = eval(collactor_cls)(
        event_tokenizer=event_tokenizer, user_tokenizer=user_tokenizer
    )

loader = DataLoader(beh_dataset, batch_size=16, collate_fn = data_collator, shuffle=True)
optimizer = AdamW(tab_net.model.parameters(), lr=0.0001)

for epoch in range(3):
    for batch in loader:

        """
            batch: [bsz, n_cols]
            event_sequence_outputs: [bsz, ncols, hidden_size]
        """

    
        import pdb
        pdb.set_trace()


