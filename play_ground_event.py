from dataset.event import EventDataset
from dataset.user import UserDataset
from dataset.vocab import load_vocab
from transformers import BertTokenizer
from dataset.datacollator import EventDataCollatorForLanguageModeling, UserDataCollatorForLanguageModeling

from models.modules import TabFormerBertLM
from torch.utils.data import DataLoader


dataset = EventDataset(mlm=True,
                 estids=None,
                 seq_len=6,
                 cached=False,
                 root="./data/event_w_dedupe/",
                 fname="event_data",
                 vocab_dir="vocab",
                 fextension="event_w_dedupe",
                 nrows=None,
                 flatten=False,
                 stride=2,
                 skip_user=True)

vocab = dataset.vocab

custom_special_tokens = vocab.get_special_tokens()
tokenizer = BertTokenizer(vocab_file="vocab/vocab_event_w_dedupe.nb", do_lower_case=False, **custom_special_tokens)
collactor_cls = "EventDataCollatorForLanguageModeling"
import pdb
pdb.set_trace()
tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=True,
                               flatten=False,
                               ncols=dataset.ncols,
                               field_hidden_size=64
                               )
                               
data_collator = eval(collactor_cls)(tokenizer = tab_net.tokenizer)

loader = DataLoader(dataset, batch_size=16, collate_fn = data_collator, shuffle=True)

for batch in loader:
    """
        batch: [bsz, n_cols]
        event_sequence_outputs: [bsz, ncols, hidden_size]
    """
    import pdb
    pdb.set_trace()
