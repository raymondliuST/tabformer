import argparse


def define_main_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument("--seed", type=int,
                        default=12,
                        help="seed to use: 9[default]")

    parser.add_argument("--mlm_prob", type=float,
                        default=0.35,
                        help="mask mlm_probability")

    parser.add_argument("--mode", type=str,
                        default='user',
                        help="training mode")
    parser.add_argument("--output_dir", type=str,
                        default='checkpoints',
                        help="path to model dump")
    parser.add_argument("--checkpoint", type=int,
                        default=0,
                        help='set to continue training from checkpoint')
    parser.add_argument("--do_train", action='store_true',
                        help="enable training flag")
    parser.add_argument("--do_eval", action='store_true',
                        help="enable evaluation flag")
    parser.add_argument("--save_steps", type=int,
                        default=500,
                        help="set checkpointing")
    parser.add_argument("--num_train_epochs", type=int,
                        default=25,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=256,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=int,
                        default=4e-04,
                        help="number of training epochs")
    
    parser.add_argument("--field_hs", type=int,
                        default=768,
                        help="hidden size for transaction transformer")

    return parser
