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
from models.tabformer_bart import TabFormerBartForMaskedLM, TabFormerBartConfig

from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling, BehaviorDataCollatorForLanguageModeling

import pandas as pd

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


beh_dataset = BehaviorDataset(
                 mlm=False,
                 estids=None,
                 seq_len=6,
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
import pdb
pdb.set_trace()
from models.modules import TabFormerHierarchicalLM
from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel
)

user_model_config = TabFormerBertConfig.from_pretrained("output/user/checkpoint-2500")
user_model = TabFormerBertForMaskedLM(user_model_config, user_vocab)
user_model = user_model.from_pretrained("output/user/checkpoint-2500", config=user_model_config, vocab = user_vocab)

event_model_config = TabFormerBertConfig.from_pretrained("output/event/checkpoint-500")
event_model = TabFormerHierarchicalLM(event_model_config, event_vocab)
event_model = event_model.from_pretrained("output/event/checkpoint-500", config=event_model_config, vocab = event_vocab)


# tokenizer = BertTokenizer(event_vocab.filename, do_lower_case=False, **custom_special_tokens)
# collactor_cls = "TransDataCollatorForLanguageModeling"
# data_collator = eval(collactor_cls)(
#         tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#     )

# loader = DataLoader(event_dataset, batch_size=16, collate_fn = data_collator, shuffle=True)

# for batch in loader:
#     """
#         batch: [bsz, seq_len, n_cols]
#         event_sequence_outputs: [bsz, (seq_len * ncols), hidden_size]
#     """
#     event_outputs = event_model(**batch)

# collactor_cls = "UserDataCollatorForLanguageModeling"
# data_collator = eval(collactor_cls)(
#         tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#     )

# loader = DataLoader(user_dataset, batch_size=16, collate_fn = data_collator, shuffle=True)

# for batch in loader:
#     """
#         batch: [bsz, n_cols]
#         event_sequence_outputs: [bsz, ncols, hidden_size]
#     """
#     user_outputs = user_model(**batch)

    

event_tokenizer = BertTokenizer(event_vocab.filename, do_lower_case=False, **custom_special_tokens)
user_tokenizer = BertTokenizer(user_vocab.filename, do_lower_case=False, **custom_special_tokens)
collactor_cls = "BehaviorDataCollatorForLanguageModeling"
data_collator = eval(collactor_cls)(
        event_tokenizer=event_tokenizer, user_tokenizer=user_tokenizer
    )

loader = DataLoader(beh_dataset, batch_size=16, collate_fn = data_collator, shuffle=True)

for batch in loader:
    """
        batch: [bsz, n_cols]
        event_sequence_outputs: [bsz, ncols, hidden_size]
    """
    import pdb
    pdb.set_trace()

