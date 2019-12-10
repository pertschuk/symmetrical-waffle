from collections import defaultdict

RERANK_NUM = 1000


def pad_passages(passages):
  while len(passages) < RERANK_NUM:
    passages += (passages[0][0], 'FAKE PASSAGE', 0.0)
  return passages


# create dev subset of 100 queries = 100,000 to rank
def main():
  qrels = []
  with open('./qrels.dev.small.tsv', 'r') as qrels_file:
    for line in qrels_file:
      qid, _, cid, _ = line.rstrip().split('\t')
      qrels.append((qid, cid))

  dev_set = defaultdict(list)
  j = 0
  with open('./top1000.dev', 'r') as top1000_dev:
    for line in top1000_dev:
      qid, cid, query, passage = line.rstrip().split('\t')
      label = 1 if (qid, cid) in qrels else 0
      dev_set[qid].append((query, passage, label))
      j += 1
      if j > 10000: break

  with open('./test_set.dev', 'w') as test_set:
    i = 0
    for qid, passages in dev_set.items():
      passages = pad_passages(passages)
      try:
        for (query, passage, label) in passages:
          test_set.write(query + '\t' + passage + '\t' + str(label) + '\n')
      except:
        import pdb
        pdb.set_trace()
      i += 1
      if i > 100: break


if __name__ == '__main__':
  main()