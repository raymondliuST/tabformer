from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        
        log_dict = {}
        for field_name, acc_dict in outputs["metric_dict"].items():
            log_dict.update({f"train_step_{metric_name}_{field_name}": acc.item() for metric_name, acc in acc_dict.items()})
        self.log(log_dict)

        # TODO: need to find a place to log eval metrics. Compute loss isn't called during evalute
        # else:
        #     log_dict = {}
        #     for field_name, acc_dict in outputs["metric_dict"].items():
        #         log_dict.update({f"val_step_{metric_name}_{field_name}": acc.item() for metric_name, acc in acc_dict.items()})
        #     self.log(log_dict)

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

