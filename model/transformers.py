import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch.nn
import torch
from nboost.model.base import BaseModel
from model.bert_model import tokenization
from collections import defaultdict


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
        self.vocab_file = str(self.model_dir.joinpath('vocab.txt'))
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        query = tokenization.convert_to_unicode(str(query))
        query_token_ids = tokenization.convert_to_bert_input(
            text=query, max_seq_length=self.max_seq_len, tokenizer=tokenizer,
            add_cls=True)
        all_features = []

        for i, choice in enumerate(choices):
            doc_text = str(choice.body)
            doc_token_id = tokenization.convert_to_bert_input(
                text=tokenization.convert_to_unicode(doc_text),
                max_seq_length=self.max_seq_len - len(query_token_ids),
                tokenizer=tokenizer,
                add_cls=False)

            query_ids = query_token_ids
            doc_ids = doc_token_id
            input_ids = query_ids + doc_ids

            query_segment_id = [0] * len(query_ids)
            doc_segment_id = [1] * len(doc_ids)
            segment_ids = query_segment_id + doc_segment_id

            input_mask = [1] * len(input_ids)

            features = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "attention_mask": input_mask,
            }
            all_features.append(features)

            def to_tsv(name):
                return ','.join([str(f) for f in features[name]])
            with open('pt_features.txt', 'a') as tf_features:
                tf_features.write(doc_text + '\t' + to_tsv('input_ids') + '\t'
                                  + to_tsv('segment_ids') + '\n')

        max_len = min(max(len(t['input_ids']) for t in all_features), self.max_seq_len)
        batches = defaultdict(list)
        for features in all_features:
            for k, v in features.items():
                batches[k].append(v + [0] * (max_len - len(v[:max_len])))

        for k, v in batches.items():
            batches[k] = torch.tensor(v).to(self.device, non_blocking=True)

        return batches['input_ids'], batches['attention_mask'], batches['token_type_ids']

    def __exit__(self, *args):
        self.rerank_model = None

    def __enter__(self, *args):
        return self
