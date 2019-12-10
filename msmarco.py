import argparse
import numpy as np
from tqdm import tqdm


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
    for line in eval_iterator:
      query, passage, label = line.rstrip().split('\t')
      candidates.append(passage)
      labels.append(int(label))
      i += 1
      if i % args.rerank_num == 0:
        if sum(labels) == 0: continue
        total += 1
        print('ranking %s' % len(candidates))
        print(candidates)
        ranks = model.rank(query, candidates)
        total_mrr += 1/(np.sum(np.array(labels) * ranks) + 1)
        eval_iterator.set_description("Current rank: %s" % ranks[np.argmax(labels)] +
                                      " MRR: %s" % (total_mrr / total) + "Total: %s " % len(candidates))
        candidates = []
        labels = []


def main():
  if args.model_class == 'bert_model':
    from bert_model import BertModel
    model = BertModel()
  else:
    from transformers_model import TransformersModel
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