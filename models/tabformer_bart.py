from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right, BartPreTrainedModel
from transformers.models.bart.configuration_bart import BartConfig
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
logger = logging.get_logger(__name__)


class TabFormerBartPredictionHead(nn.Module):
    def __init__(self, config):
        super(TabFormerBartPredictionHead, self).__init__()
        self.d_model = config.d_model
        self.n_cols = config.decoder_ncols
        self.vocab_size = config.decoder_vocab_size
        
        self.fc = nn.Linear(self.d_model, self.d_model) # Example fully connected layer
        self.fc_out = nn.Linear(self.d_model, self.n_cols * self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            input:
                x: (bsz, seq_len, d_model) the outputs from transformer decoder

            output:
                x: (bsz, seq_len, n_cols, vocab_size)
        """
        # Reshape input tensor
        bsz, seq_len, _ = x.size()
        x = x.view(-1, self.d_model)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc(x))
        x = self.fc_out(x)
        
        # Reshape back
        x = x.view(bsz, seq_len, self.n_cols, self.vocab_size)
        
        return x


class TabFormerBartConfig(BartConfig):
    def __init__(
        self,
        encoder_ncols=10,
        decoder_ncols=6,
        encoder_vocab_size=500,
        decoder_vocab_size=500,
        encoder_layers=1,
        encoder_ffn_dim=60,
        encoder_attention_heads=10,
        decoder_layers=1,
        decoder_ffn_dim=36,
        decoder_attention_heads=6,
        pad_token_id=0,
        d_model=60,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.encoder_ncols=encoder_ncols
        self.decoder_ncols=decoder_ncols
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_ffn_dim = encoder_ffn_dim

        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
    
        self.d_model = d_model


class HierarchicalEncoder(nn.Module):
    """ HierarchicalEncoder: Encodes the tabular decoder input into decoder embeddings 
                            using multivariate time serie transformer encoder
        
        Args:
            config.decoder_ncols
            config.decoder_vocab_size
            config.hidden_size
            config.decoder_ffn_dim
            config.decoder_attention_heads
            config.decoder_layers (int): Number of transformer layers

         Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.decoder_vocab_size, config.d_model,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.decoder_attention_heads,
                                                   dim_feedforward=config.d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.decoder_layers)

        self.lin_proj = nn.Linear(config.d_model * config.decoder_ncols, config.hidden_size)

    def forward(self, input_ids):

        inputs_embeds = self.word_embeddings(input_ids)

        embeds_shape = list(inputs_embeds.size()) # (bsz, seq_len, ncols, d_model)
        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:]) # ((bsz*seq_len), ncols, d_model)
        inputs_embeds = inputs_embeds.permute(1, 0, 2) # (ncols, (bsz*seq_len), d_model)
        
        inputs_embeds = self.transformer_encoder(inputs_embeds) # (ncols, (bsz*seq_len), d_model)

        inputs_embeds = inputs_embeds.permute(1, 0, 2)  # (ncols, (bsz*seq_len), d_model)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2]+[-1]) # (ncols, (bsz*seq_len), (d_model*bnheads))

        inputs_embeds = self.lin_proj(inputs_embeds) # (ncols, (bsz*seq_len), d_model)

        return inputs_embeds

class TabFormerBart(BartForConditionalGeneration):
    def __init__(self, config, encoder_vocab, decoder_vocab):
        super().__init__(config)

        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab

        self.decoder_input_encoder = HierarchicalEncoder(config)

        self.cls = TabFormerBartPredictionHead(config)

        self.post_init()  

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):


        """
            Args:
                input_ids: (bsz, input_ncols)
                    input to encoder, i.e user data
        """
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Insert the <start> token
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        decoder_inputs_embeds = self.decoder_input_encoder(decoder_input_ids) # (bsz, seq_len, d_model)
        decoder_input_ids = None

        outputs = self.model(
            input_ids, # encoder input 
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, # None because we use embedding
            encoder_outputs=encoder_outputs, # None because we use encoder_input_ids
            decoder_attention_mask=decoder_attention_mask, # the attention mask on events after padding
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * d_model]

        prediction_scores = self.cls(sequence_output) # [bsz, seq_len, n_cols, d_model]

        outputs = (prediction_scores,) + outputs[2:]

        total_masked_lm_loss = 0
        field_names = self.decoder_vocab.get_field_keys(ignore_special=True)

        seq_len = prediction_scores.size(1)
        for field_idx, field_name in enumerate(field_names):
            
            global_ids_field = self.decoder_vocab.get_field_ids(field_name)
            nfeas = len(global_ids_field)

            prediction_scores_field = prediction_scores[:, :, field_idx, global_ids_field]  # [bsz , seq_len , field_vocab_size]
            labels_field_global = labels[:,:,field_idx] # [bsz, seq_len]
            labels_field_local = self.decoder_vocab.get_from_global_ids(global_ids=labels_field_global,
                                                                        what_to_get='local_ids')
  
            loss_fct = CrossEntropyLoss(reduction='mean')
            lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)), labels_field_local.view(-1))
            
            total_masked_lm_loss += lm_loss_field


        return  {"loss": total_masked_lm_loss, 
                "outputs": outputs, 
                "metric_dict": {}}
         
            
            

