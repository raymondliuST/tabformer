from models.hierarchical import TabFormerHierarchicalLM
from models.tabformer_bart import TabFormerBart, TabFormerBartConfig
from models.tabformer_bert import TabFormerBertConfig, TabFormerBertForMaskedLM
from transformers import BertTokenizer 


class PeopleModelModule:
    def __init__(self, 
                mode="user",
                special_tokens=None, 
                user_vocab=None, 
                event_vocab=None, 
                user_ncols=None, 
                event_ncols=None, 
                field_hidden_size=768):

        
        self.user_vocab = user_vocab
        self.user_ncols = user_ncols
        self.event_vocab = event_vocab
        self.event_ncols = event_ncols

        if mode == "user":
            vocab_file = self.user_vocab.filename
            hidden_size = field_hidden_size
            self.config = TabFormerBertConfig(mode="user",
                                            vocab_size=len(self.user_vocab),
                                            ncols=self.user_ncols,
                                            hidden_size=hidden_size,
                                            field_hidden_size=field_hidden_size,
                                            num_attention_heads=self.user_ncols)

            self.tokenizer = BertTokenizer(vocab_file,
                                    do_lower_case=False,
                                    **special_tokens)
        elif mode == "event":
            vocab_file = self.event_vocab.filename
            hidden_size = field_hidden_size * self.event_ncols
            self.config = TabFormerBertConfig(mode="event",
                                            vocab_size=len(self.event_vocab),
                                            ncols=self.event_ncols,
                                            hidden_size=hidden_size,
                                            field_hidden_size=field_hidden_size,
                                            num_attention_heads=self.event_ncols)

            self.tokenizer = BertTokenizer(vocab_file,
                                    do_lower_case=False,
                                    **special_tokens)
        elif mode == "behavior":
            vocab_file = self.user_vocab.filename
            hidden_size = field_hidden_size

            pad_token_id=self.event_vocab.get_id(event_vocab.pad_token, special_token=True)
            bos_token_id=self.event_vocab.get_id(event_vocab.bos_token_id, special_token=True)
            eos_token_id=self.event_vocab.get_id(event_vocab.eos_token_id, special_token=True)

            self.config = TabFormerBartConfig(
                encoder_vocab_size=len(user_vocab),
                decoder_vocab_size=len(event_vocab),
                encoder_ncols=self.user_ncols,
                decoder_ncols=self.event_ncols,
                d_model=60,
                dropout=0.1,
                field_hidden_size=64,
                encoder_layers=1,
                encoder_ffn_dim=128,
                encoder_attention_heads=self.user_ncols,
                decoder_layers=1,
                decoder_ffn_dim=128,
                decoder_attention_heads=self.event_ncols,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                decoder_start_token_id=eos_token_id,
                )
        else:
            raise Exception("Mode not supported")

        self.model = self.get_model(mode)

    def get_model(self, mode):
        if mode == "user":
            # flattened BERT
            model = TabFormerBertForMaskedLM(self.config, self.user_vocab, mode='user')
        elif mode == "event":
            # hierarchical BERT
            model = TabFormerHierarchicalLM(self.config, self.event_vocab, mode='event')
        else:
            model = TabFormerBart(self.config, self.user_vocab, self.event_vocab)

        return model