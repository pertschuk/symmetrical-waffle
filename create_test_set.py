from collections import defaultdict

def main():
  dev_set = defaultdict(list)
  with open('./test_set.dev', 'w') as pt_test_set:
    with open('./top1000.dev', 'r') as top1000_dev:
      for line in top1000_dev:
        qid, doc_id, query, passage = line.rstrip().split('\t')
        pt_test_set.write(query + '\t' + passage + '\n')

if __name__ == '__main__':
  main()