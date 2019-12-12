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
      labels.append(int(float(label)))
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
        ranks, logits = model.rank(query.encode(), choices)
        top_rank = np.argmax(np.array(labels)[ranks])
        total_mrr += 1/(top_rank + 1)
        eval_iterator.set_description("Current rank: %s" % top_rank +
                                      " MRR: %s" % (total_mrr / total) + "Total: %s " % len(candidates))
        candidates = []
        labels = []
        queries = []


def test_equivilency():
  from model.transformers import TransformersModel
  from model.bert_model import BertModel
  tf_model = BertModel(model_dir=args.tf_model, batch_size=args.batch_size)
  pt_model = TransformersModel(model_dir=args.pt_model, batch_size=args.batch_size)
  with open('test_set.tsv', 'r') as test_set:
    for line in test_set:
      query, passage, label = line.rstrip().split('\t')
      choices = [Choice('0', passage)]
      _, tf_logits = tf_model.rank(query.encode(), choices)
      _, pt_logits = pt_model.rank(query.encode(), choices)
      try:
        np.testing.assert_allclose(tf_logits, pt_logits)
      except:
        import pdb
        pdb.set_trace()

def main():
  if args.test_eq:
    test_equivilency()
    return
  if args.model_class == 'bert_model':
    from model.bert_model import BertModel
    model = BertModel(model_dir=args.tf_model, batch_size=args.batch_size)
  else:
    from model.transformers import TransformersModel
    model = TransformersModel(model_dir=args.pt_model, batch_size=args.batch_size)
  eval(model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--eval_steps', default=1000, type=int)
  parser.add_argument('--tf_model', default='bert-base-uncased-msmarco')
  parser.add_argument('--pt_model', default='pt-bert-base-uncased-msmarco')
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--max_length', default=128, type=int)
  parser.add_argument("--model_class", default='bert_model')
  parser.add_argument("--rerank_num", default=1000, type=int)
  parser.add_argument('--test_eq', action='store_true')
  args = parser.parse_args()
  main()