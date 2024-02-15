from os import makedirs
from os.path import join
import os
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import BertTokenizer

from transformers import TrainingArguments
from custom_trainer import CustomTrainer
from dataset.behavior import BehaviorDataset
from dataset.vocab import load_vocab
from models.modules import PeopleModelModule
from misc.utils import random_split_dataset
from dataset.datacollator import BehaviorDataCollatorForLanguageModeling, EventDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling
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

    user_vocab = None
    event_vocab = None
    user_ncols = None
    event_ncols = None
    if args.mode == "user":
        dataset = UserDataset(mlm=True,
                 estids=None,
                 cached=True,
                 root="./data/user/",
                 fname="user_data",
                 vocab_dir="vocab",
                 fextension="user",
                 flatten=True,
                 skip_user=True)
        user_vocab = dataset.vocab
        user_ncols = dataset.ncols
        custom_special_tokens = user_vocab.get_special_tokens()

        user_tokenizer = BertTokenizer(vocab_file=user_vocab.filename, do_lower_case=False, **custom_special_tokens)
        data_collator = UserDataCollatorForLanguageModeling(tokenizer=user_tokenizer, 
                                                            mlm=True, 
                                                            mlm_probability=args.mlm_prob)
    elif args.mode == 'event':
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
                 skip_user=True)
        event_vocab = dataset.vocab
        event_ncols = dataset.ncols
        custom_special_tokens = event_vocab.get_special_tokens()

        event_tokenizer = BertTokenizer(vocab_file=event_vocab.filename, do_lower_case=False, **custom_special_tokens)
        data_collator = EventDataCollatorForLanguageModeling(tokenizer=event_tokenizer, 
                                                            mlm=True, 
                                                            mlm_probability=args.mlm_prob)
    else:
        user_vocab = load_vocab("vocab/vocab_user.json")
        event_vocab = load_vocab("vocab/vocab_event_w_dedupe.json")
        
        dataset = BehaviorDataset(mlm=False,
                 estids=None,
                 cached=True,
                 root="./data/behavior/",
                 fname="people-model-20240124",
                 fextension="behavior",
                 nrows=None,
                 flatten=True, #TODO: delete flatten
                 skip_user=True,
                 user_vocab = user_vocab,
                 event_vocab = event_vocab)
        user_ncols = dataset.user_ncols
        event_ncols = dataset.event_ncols

        custom_special_tokens = event_vocab.get_special_tokens()
        collactor_cls = "UserDataCollatorForLanguageModeling"

        user_tokenizer = BertTokenizer(vocab_file=user_vocab.filename, do_lower_case=False, **custom_special_tokens)
        event_tokenizer = BertTokenizer(vocab_file=event_vocab.filename, do_lower_case=False, **custom_special_tokens)
        data_collator = BehaviorDataCollatorForLanguageModeling(event_tokenizer=event_tokenizer, 
                                                                user_tokenizer=user_tokenizer)

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


    # loading model
    tab_net = PeopleModelModule(mode=args.mode,
                                special_tokens=custom_special_tokens,
                                user_vocab=user_vocab,
                                event_vocab=event_vocab,
                                user_ncols=user_ncols,
                                event_ncols=event_ncols,
                                field_hidden_size=args.field_hs)

    log.info(f"model initiated: {tab_net.model.__class__}")

    print(f"logdir: {args.log_dir}")

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
        run_name=f"{args.mode}/lr={args.learning_rate}/hs={args.field_hs}",
        dataloader_num_workers=4,
        fp16= True
    )   
   
    
    # optimizer 
    # training_args = training_args.set_optimizer(learning_rate = args.learning_rate, beta1=0.9)
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

    # if opts.mlm and opts.lm_type == "gpt2":
    #     raise Exception("Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    # if not opts.mlm and opts.lm_type == "bert":
    #     raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)
