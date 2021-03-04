import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


from torchfly.training import FlyModel
from torchfly.metrics import CategoricalAccuracy, Average, MovingAverage, Speed

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerFast

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

class KnowledgeDialogModel(nn.Module):
    def __init__(self, config):
        super(KnowledgeDialogModel, self).__init__()

        # TODO extend to models other than t5
        self.config = config

        self.model = T5ForConditionalGeneration.from_pretrained(config.model.name)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model.name)

    def forward(self, batch):
        # breakpoint()
        return self.model(input_ids=batch["context_tensor"],
                        attention_mask=batch["context_mask"], 
                        labels=batch["target_tensor"])


    def generate(self, batch):

        # breakpoint()
        generation_outputs = self.model.generate(input_ids=batch["context_tensor"],
                            num_beams=4, 
                            num_return_sequences=1,
                            max_length=self.config.training.generation.max_length)
        generated_batch = [self.tokenizer.decode(one, skip_special_tokens=True) 
                            for one in generation_outputs]
        
        labels = batch["target_tensor"].clone()
        labels[labels==-100] = 0
        label_batch = [self.tokenizer.decode(one, skip_special_tokens=True) 
                            for one in labels]

        bleu_2 = [sentence_bleu(references=[word_tokenize(one_ref)],
                                hypothesis=word_tokenize(one_candidate),
                                weights=(1.0/3, 1.0/3, 1.0/3)) 
                    for one_ref, one_candidate in zip(label_batch, generated_batch)]
        
        context_batch = [self.tokenizer.decode(one, skip_special_tokens=True) 
                            for one in batch["context_tensor"]]
        
        pairs = [i for i in zip(context_batch, label_batch, generated_batch)]

        return {"bleu-2": bleu_2, "pairs": pairs}


class KnowledgeDialogFlyModel(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = KnowledgeDialogModel(config)

    def configure_metrics(self):
        self.training_metrics = {"loss": MovingAverage(), "dialogs/s": Speed()}
        # self.evaluation_metrics = {"loss": Average(), "acc": CategoricalAccuracy()}
        self.evaluation_metrics = {"loss": Average(), "bleu-2": Average()}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        # breakpoint()
        # x, target = batch
        outputs = self.model(batch)
        loss = outputs.loss

        # generation_output = self.model.generate(batch)

        results = {"loss": loss, "output": outputs}
        self.training_metrics["loss"](loss.item())

        return results

    def predict(self, batch):
        # x, target = batch
        # breakpoint()
        outputs = self.model(batch)
        # breakpoint()
        loss = outputs.loss
        # loss = F.nll_loss(output, target)
        self.evaluation_metrics["loss"](loss.item())

        generation_scores = self.model.generate(batch)
        self.evaluation_metrics["bleu-2"](generation_scores["bleu-2"])
        
        return generation_scores["pairs"]

    def get_training_metrics(self) -> Dict[str, str]:
        loss = self.training_metrics["loss"].get_metric()
        metrics = {"loss": f"{loss:.4f}"}
        return metrics

    def get_evaluation_metrics(self) -> Dict[str, str]:
        loss = self.evaluation_metrics["loss"].get_metric()
        bleu_2 = self.evaluation_metrics["bleu-2"].get_metric()
        metrics = {"loss": f"{loss:.4f}", "bleu-2": f"{bleu_2:.4f}"}
        # metrics = {"loss": f"{loss:.4f}"}
        return metrics
        