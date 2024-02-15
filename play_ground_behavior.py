from dataset.behavior import BehaviorDataset
from dataset.event import EventDataset
from dataset.user import UserDataset
from dataset.vocab import load_vocab
from transformers import BertTokenizer
from dataset.datacollator import BehaviorDataCollatorForLanguageModeling

from models.modules import TabFormerBertLM
from torch.utils.data import DataLoader

from models.tabformer_bart import TabFormerBart, TabFormerBartConfig

user_vocab = load_vocab("vocab/vocab_user.json")
event_vocab = load_vocab("vocab/vocab_event_w_dedupe.json")

dataset = BehaviorDataset(mlm=False,
                 estids=None,
                 cached=True,
                 root="./data/behavior/",
                 fname="people-model-20240124",
                 fextension="behavior",
                 nrows=None,
                 flatten=True,
                 skip_user=True,
                 user_vocab = user_vocab,
                 event_vocab = event_vocab)

custom_special_tokens = user_vocab.get_special_tokens() # user and event vocab should have same special tokens

user_tokenizer = BertTokenizer(vocab_file="vocab/vocab_user.nb", do_lower_case=False, **custom_special_tokens)
event_tokenizer = BertTokenizer(vocab_file="vocab/vocab_event_w_dedupe.nb", do_lower_case=False, **custom_special_tokens)
collactor_cls = "BehaviorDataCollatorForLanguageModeling"
data_collator = eval(collactor_cls)(event_tokenizer = event_tokenizer, user_tokenizer = user_tokenizer)

pad_token = event_vocab.get_id(event_vocab.pad_token, special_token=True)
bos_token = event_vocab.get_id(event_vocab.bos_token, special_token=True)  # will be used for GPT2
eos_token = event_vocab.get_id(event_vocab.eos_token, special_token=True)

config = TabFormerBartConfig(
        encoder_vocab_size=len(user_vocab),
        decoder_vocab_size=len(event_vocab),
        encoder_ncols=dataset.user_ncols,
        decoder_ncols=dataset.event_ncols,
        d_model=60,
        dropout = 0.1,
        field_hidden_size=64,
        encoder_layers=1,
        encoder_ffn_dim=128,
        encoder_attention_heads=dataset.user_ncols,
        decoder_layers=1,
        decoder_ffn_dim=128,
        decoder_attention_heads=dataset.event_ncols,
        pad_token_id=pad_token,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
        decoder_start_token_id=eos_token
        )

tab_net = TabFormerBart(config, user_vocab, event_vocab)

loader = DataLoader(dataset, batch_size=16, collate_fn = data_collator, shuffle=True)

for batch in loader:
    """
        batch: [bsz, n_cols]
        event_sequence_outputs: [bsz, ncols, hidden_size]
    """
    import pdb
    pdb.set_trace()
