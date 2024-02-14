from dataset.event import EventDataset
from dataset.user import UserDataset
from dataset.vocab import load_vocab

user_dataset = UserDataset(mlm=True,
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

event_dataset = EventDataset(mlm=True,
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
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=True)



    


