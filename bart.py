from os import makedirs
from os.path import join
import os
import logging
import numpy as np
import torch
import random
from args import define_main_parser


from torch.utils.data import DataLoader
from dataset.event import EventDataset
from dataset.user import UserDataset
from dataset.behavior import BehaviorDataset


from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from custom_trainer import CustomTrainer
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling, BehaviorDataCollatorForLanguageModeling
from dataset.event import EventDataset
from dataset.user import UserDataset
import wandb

from models.tabformer_bart import TabFormerBart, TabFormerBartConfig 

from transformers import (
    BertTokenizer,
)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

wandb.init()
logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

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

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(beh_dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(beh_dataset, lengths)
 
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

    event_tokenizer = BertTokenizer(event_vocab.filename, do_lower_case=False, **custom_special_tokens)
    user_tokenizer = BertTokenizer(user_vocab.filename, do_lower_case=False, **custom_special_tokens)
    data_collator = eval("BehaviorDataCollatorForLanguageModeling")(
        event_tokenizer=event_tokenizer, user_tokenizer=user_tokenizer
    )


    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=args.log_dir,  # directory for storing logs
        save_steps=args.save_steps,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="epoch",
        prediction_loss_only=True,
        overwrite_output_dir=True,
        logging_steps=100,
        report_to="wandb",
        run_name=f"{args.data_type}/lr={args.learning_rate}/hs={args.field_hs}",
        dataloader_num_workers=4,
        fp16= True,
        remove_unused_columns=False
    )   
    
    # optimizer 
    # training_args = training_args.set_optimizer(learning_rate = args.learning_rate, beta1=0.9)
    print(training_args.num_train_epochs)

    trainer = Trainer(
        model=tab_net,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    if args.checkpoint:
        model_path = join(args.output_dir, f'checkpoint-{args.checkpoint}')
    else:
        model_path = args.output_dir

    trainer.train()


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    main(opts)