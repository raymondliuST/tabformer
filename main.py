from os import makedirs
from os.path import join
import os
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from custom_trainer import CustomTrainer
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling
from dataset.event import EventDataset
from dataset.user import UserDataset
import wandb 

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

    if args.data_type == 'event':
        dataset = EventDataset(mlm=True,
                 estids=None,
                 seq_len=6,
                 cached=True,
                 root="./data/event_no_dedupe/",
                 fname="event_data",
                 vocab_dir="vocab",
                 fextension="no_dedupe",
                 nrows=None,
                 flatten=False,
                 stride=2,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=True)
    elif args.data_type == "user":
        dataset = UserDataset(mlm=True,
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
                 skip_user=True)
    
    vocab = dataset.vocab

    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=args.field_ce,
                               flatten=args.flatten,
                               ncols=dataset.ncols,
                               field_hidden_size=args.field_hs
                               )
    else:
        tab_net = TabFormerGPT2(custom_special_tokens,
                             vocab=vocab,
                             field_ce=args.field_ce,
                             flatten=args.flatten,
                             )

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.flatten:
        collactor_cls = "UserDataCollatorForLanguageModeling"
    else:
        collactor_cls = "TransDataCollatorForLanguageModeling"

    log.info(f"collactor class: {collactor_cls}")
    data_collator = eval(collactor_cls)(
        tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
    )
    print(f"logdir: {args.log_dir}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

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
        fp16= True
    )   

   
    
    # optimizer 
    training_args = training_args.set_optimizer(name="adamw_torch", learning_rate = args.learning_rate, beta1=0.9)
    # training_args = training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05, max_steps = -1)
    print(training_args.num_train_epochs)

    trainer = CustomTrainer(
        model=tab_net.model,
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

    if opts.mlm and opts.lm_type == "gpt2":
        raise Exception("Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)
