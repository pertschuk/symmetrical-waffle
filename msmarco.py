import argparse
import numpy as np
from tqdm import tqdm
from nboost.types import Choice


def eval(model):
  # load_and_cache_eval()
  qrels = []

  i = 0
  total = 0
  total_mrr = 0
  with open('test_set.tsv', 'r') as test_set:
    eval_iterator = tqdm(test_set, desc="Evaluating")
    candidates = []
    labels = []
    queries = []
    for line in eval_iterator:
      query, passage, label = line.rstrip().split('\t')
      queries.append(query)
      candidates.append(passage)
      labels.append(int(label))
      i += 1
      if i % args.rerank_num == 0:
        if sum(labels) == 0:
          candidates = []
          labels = []
          queries = []
          continue
        assert len(set(queries)) == 1
        total += 1
        print('ranking %s' % len(candidates))
        choices = [Choice('0', candidate) for candidate in candidates]
        ranks = model.rank(query.encode(), choices)
        total_mrr += 1/(np.sum(np.array(labels) * ranks) + 1)
        eval_iterator.set_description("Current rank: %s" % np.argmax(np.array(labels)[ranks]) +
                                      " MRR: %s" % (total_mrr / total) + "Total: %s " % len(candidates))
        candidates = []
        labels = []
        queries = []


def main():
  if args.model_class == 'bert_model':
    from nboost.model.bert_model import BertModel
    model = BertModel()
  else:
    from nboost.model.transformers import TransformersModel
    model = TransformersModel()
  eval(model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--eval_steps', default=1000, type=int)
  parser.add_argument('--model', default='bert-base-uncased-msmarco')
  parser.add_argument('--batch_size', default=8, type=int)
  parser.add_argument('--max_length', default=128, type=int)
  parser.add_argument("--model_class", default='bert_model')
  parser.add_argument("--rerank_num", default=1000, type=int)
  args = parser.parse_args()
  main()