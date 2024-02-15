import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers.models.bert.modeling_bert import ACT2FN

from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig

BertLayerNorm = torch.nn.LayerNorm

class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        mode="user",
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        hidden_size=768,
        num_attention_heads=12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.mode = "event"
        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_attention_heads=num_attention_heads

class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab, mode):
        super().__init__(config)

        self.mode = mode
        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()
        self.step = 0

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        self.step += 1
        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        if self.mode == "event":
            # need to flatten event
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1]*self.config.ncols, -1] #bsz, (seqlen * ncol), -1

            sequence_output = sequence_output.view(expected_sz) #bsz, (seqlen * ncol), hidden_size
            masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1) # bsz, (seqlen * ncol)


        prediction_scores = self.cls(sequence_output) # [bsz , (seqlen * ncols) , vocab_sz]

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0
        metric_dict = {}

        seq_len = prediction_scores.size(1)
        field_names = self.vocab.get_field_keys(ignore_special=True)

        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))
            
            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * seq_len * field_vocab_size

            masked_lm_labels_field = masked_lm_labels[:, col_ids] # bsz * (seq length) 
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                        what_to_get='local_ids')

            nfeas = len(global_ids_field)
            
            cls_weights = self.vocab.column_weights.get(field_name)
            if cls_weights is None:
                cls_weights_tensor = None
            else:
                cls_weights_tensor = torch.tensor(cls_weights).to(self.device) 
            loss_fct = CrossEntropyLoss(weight=cls_weights_tensor, reduction='mean')

            masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)), masked_lm_labels_field_local.view(-1))

            # metric
            metrics = self.compute_metrix(prediction_scores_field.view(-1, len(global_ids_field)), masked_lm_labels_field_local.view(-1), nfeas)

            total_masked_lm_loss += masked_lm_loss_field
            metric_dict[field_name] = metrics
            
        result = {"loss": total_masked_lm_loss, "outputs": outputs, "metric_dict": metric_dict}
            
        return result

    def compute_metrix(self, prediction_scores_field, masked_lm_labels_field_local, nfeas):
        micro_acc = MulticlassAccuracy(average = 'micro', ignore_index=-100, num_classes=nfeas).to(self.device) # Sum statistics over all labels
        macro_acc = MulticlassAccuracy(average = 'macro', ignore_index=-100, num_classes=nfeas).to(self.device) # Calculate statistics for each label and average them
        f1_score = MulticlassF1Score(average = 'macro', ignore_index=-100, num_classes=nfeas).to(self.device)

        return {
            "micro":micro_acc(prediction_scores_field, masked_lm_labels_field_local),
            "macro":macro_acc(prediction_scores_field, masked_lm_labels_field_local),
            "f1":f1_score(prediction_scores_field, masked_lm_labels_field_local)
        }

class TabFormerBertModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        return sequence_output


