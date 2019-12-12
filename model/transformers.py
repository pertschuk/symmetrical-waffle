import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch.nn
import torch
from nboost.model.base import BaseModel


class TransformersModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download()

        self.logger.info('Loading from checkpoint %s' % str(self.model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == torch.device("cpu"):
            self.logger.info("RUNNING ON CPU")
        else:
            self.logger.info("RUNNING ON CUDA")
            torch.cuda.synchronize(self.device)

        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.rerank_model.to(self.device, non_blocking=True)

    def rank(self, query, choices):
        input_ids, attention_mask, token_type_ids = self.encode(query, choices)

        with torch.no_grad():
            logits = self.rerank_model(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            scores = np.squeeze(log_probs.detach().cpu().numpy())
            if len(scores.shape) > 1 and scores.shape[1] == 2:
                scores = np.squeeze(scores[:,1])
            if len(scores) == 1:
                scores = [scores]
            return np.argsort(scores)[::-1], log_probs

    def encode(self, query, choices):
        inputs = [self.tokenizer.encode_plus(
            str(query), str(choice.body), add_special_tokens=True) for choice in choices]

        def to_tsv(name, input):
            return ','.join(input[name])

        with open('pt_features.txt', 'w+') as tf_features:
            for input in inputs:
                tf_features.write(to_tsv('input_ids', input) + '\t'
                                  + to_tsv('token_type_ids', input) + '\n')

        max_len = min(max(len(t['input_ids']) for t in inputs), self.max_seq_len)
        input_ids = [t['input_ids'][:max_len] +
                     [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
        attention_mask = [[1] * len(t['input_ids'][:max_len]) +
                          [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
        token_type_ids = [t['token_type_ids'][:max_len] +
                     [0] * (max_len - len(t['token_type_ids'][:max_len])) for t in inputs]

        input_ids = torch.tensor(input_ids).to(self.device, non_blocking=True)
        attention_mask = torch.tensor(attention_mask).to(self.device, non_blocking=True)
        token_type_ids = torch.tensor(token_type_ids).to(self.device, non_blocking=True)


        return input_ids, attention_mask, token_type_ids

    def __exit__(self, *args):
        self.rerank_model = None

    def __enter__(self, *args):
        pass
