from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator
from collections.abc import Mapping
import numpy as np

def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0), None

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    if len(examples[0].shape) == 2: 
        # padding for not flatten case
        result = examples[0].new_full([len(examples), max_length, examples[0].shape[1]], tokenizer.pad_token_id) #(bsz, max_length, ncols)
        
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0], :] = example
            else:
                result[i, -example.shape[0] :] = example
        
        # - 1 for tokens that are **not masked**,
        # - 0 for tokens that are **masked**.
        attention_mask = (result[:,:,0] != tokenizer.pad_token_id).int()
    else:
        result = examples[0].new_full([len(examples), max_length, ], tokenizer.pad_token_id) #(bsz, max_length, ncols)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example

        attention_mask = (result[:,0] != tokenizer.pad_token_id).int()
    return (result, attention_mask)

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

class BehaviorDataCollatorForLanguageModeling(DefaultDataCollator):
    def __init__(self, event_tokenizer, user_tokenizer):
        super().__init__()
        self.event_tokenizer = event_tokenizer
        self.user_tokenizer = user_tokenizer

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        input_list = [f["input"] for f in features]
        label_list = [f["label"] for f in features]

        input_ids, input_attention_mask = _collate_batch(input_list, self.user_tokenizer, pad_to_multiple_of=None)
        batch = {
            "input_ids": input_ids,
            "attention_mask": input_attention_mask
        }

        label_batch, label_attention_mask = _collate_batch(label_list, self.event_tokenizer, pad_to_multiple_of=None)
        label_batch[label_batch == self.event_tokenizer.pad_token_id] = -100

        batch["labels"] = label_batch
        batch["decoder_attention_mask"] = label_attention_mask

        return batch


class EventDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        batch, attention_mask = _collate_batch(examples, self.tokenizer) # From list of tensors to tensors

        sz = batch.shape
        if self.mlm:
            # inputs, labels = self.mask_tokens(batch.view(sz[0], -1))
            inputs, labels = self.improved_mask_tokens(batch)
            output = {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
            return output
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def improved_mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        labels = inputs.clone() # bsz, seq len, ncols

        bsz, seq_len, ncols = labels.shape
        for bidx, batch in enumerate(labels):
            selected_masked_values = np.array([], dtype=int)
            for c in range(ncols):
                distinct_column_values = np.unique(batch[:, c].tolist())

                col_mask = np.random.binomial(1, self.mlm_probability, size=distinct_column_values.shape).astype(bool)

                # Apply mask to select values
                selected_masked_value = distinct_column_values[col_mask]
                selected_masked_values = np.append(selected_masked_values, selected_masked_value)

            batch_mask = torch.isin(batch, torch.from_numpy(selected_masked_values))
            labels[bidx][~batch_mask] = -100

        masked_indices = labels != -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels


    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



class UserDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        collated_batch = _collate_batch(examples, self.tokenizer, pad_to_multiple_of=None)

        batch = {
            "input_ids": collated_batch[0],
            "attention_mask": collated_batch[1]
        }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            batch["input_ids"], batch["masked_lm_labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch
    