import argparse
from tqdm import tqdm, trange
import torch
from transformers import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
import pathlib
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def load_pretrained():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = BertConfig.from_pretrained(args.model)
  if not config.num_labels == 2:
    config.num_labels = 2  # regression
  model = BertForSequenceClassification.from_pretrained(args.model, config=config)
  model.to(device)
  tokenizer = BertTokenizer.from_pretrained(args.model)
  return device, model, tokenizer


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.seq_length = seq_length
    self.label_id = label_id


def inputs_to_features(inputs):
  input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
  attention_mask = [1] * len(input_ids)
  padding_length = args.max_length - len(input_ids)
  input_ids = input_ids + ([0] * padding_length)
  attention_mask = attention_mask + ([0] * padding_length)
  token_type_ids = token_type_ids + ([0] * padding_length)
  return input_ids, attention_mask, token_type_ids


def load_and_cache_eval():
  qrels = []
  device, model, tokenizer = load_pretrained()
  with open('./qrels.dev.small.tsv', 'r') as qrels_file:
    for line in tqdm(qrels_file, desc="loading qrels"):
      qid, _, cid, _ = line.rstrip().split('\t')
      qrels.append((qid, cid))

  dev_set = defaultdict(list)
  with open('./top1000.dev', 'r') as dev_file:
    for line in tqdm(dev_file, desc='loading dev file'):
      qid, cid, query, candidate = line.rstrip().split('\t')
      label = 1 if (qid, cid) in qrels else 0
      dev_set[query].append((candidate, label, qid))

  eval_iterator = tqdm(dev_set.items(), desc="Creating eval set")
  all_input_ids = []
  all_attention_masks = []
  all_token_type_ids = []
  with open('qids.tsv', 'w') as qids_file:
    for (query, choices) in eval_iterator:
      candidates = [choice[0] for choice in choices]
      labels = [choice[1] for choice in choices]
      qids = [choice[2] for choice in choices]
      if sum(labels) == 0: continue

      all_features = encode(tokenizer, query, candidates)
      for input_ids, attention_mask, token_type_ids in all_features:
        all_input_ids.extend(input_ids)
        all_attention_masks.extend(attention_mask)
        all_token_type_ids.extend(token_type_ids)
      for qid, label in zip (qids, labels):
        qids_file.write(str(qid) + '\t' + str(label) + '\n')

  dev_set = TensorDataset(
    torch.tensor(all_input_ids, dtype=torch.long),
    torch.tensor(all_attention_masks, dtype=torch.long),
    torch.tensor(all_token_type_ids, dtype=torch.long),
  )
  torch.save(dev_set, './dev_set.bin')



def load_and_cache_triples(triples_path: pathlib.Path, tokenizer):
  cache_path = triples_path.with_suffix('.bin-%s' % args.steps)

  if not cache_path.exists():
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_labels = []

    with triples_path.open('r') as f:
      for i, line in enumerate(tqdm(f, desc="Loading train triples")):
        if i * 2 > args.batch_size * args.steps:
          break
        query, relevant_example, negative_example = line.rstrip().split('\t')

        for passage in (relevant_example, negative_example):
          inputs = tokenizer.encode_plus(
            query,
            passage.lower(),
            add_special_tokens=True,
            max_length=args.max_length
          )
          # print("query: %s" % query)
          # print("passage: %s" % passage)
          input_ids, attention_mask, token_type_ids = inputs_to_features(inputs)
          all_input_ids.append(input_ids)
          all_attention_mask.append(attention_mask)
          all_token_type_ids.append(token_type_ids)
        all_labels.extend([1, 0])

    dataset = TensorDataset(
      torch.tensor(all_input_ids, dtype=torch.long),
      torch.tensor(all_attention_mask, dtype=torch.long),
      torch.tensor(all_token_type_ids, dtype=torch.long),
      torch.tensor(all_labels, dtype=torch.long)
    )
    torch.save(dataset, str(cache_path))

  else:
    dataset = torch.load(str(cache_path))

  return dataset


def train(device, model, tokenizer):
  os.makedirs(args.save_dir, exist_ok=True)

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=args.warmup * args.steps,
                                              num_training_steps=args.steps)
  fp16 = False
  try:
    from apex import amp
    fp16 = True
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
  except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

  train_dataset = load_and_cache_triples(pathlib.Path(args.triples_path), tokenizer)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

  global_step = 0
  tr_loss, logging_loss, total_pred = 0.0, 0.0, 0.0
  correct = 0
  model.zero_grad()
  epoch_iterator = tqdm(train_dataloader, desc="Iteration")
  for step, batch in enumerate(epoch_iterator):
    model.train()
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'token_type_ids': batch[2], # change for distilbert
              'labels': batch[3]}
    outputs = model(**inputs)
    loss = outputs[0]
    # total_pred += torch.sum(outputs[1]).item() / args.batch_size
    logits = outputs[1].detach().cpu().numpy()
    correct += np.sum(np.argmax(logits, axis=1) == batch[3].detach().cpu().numpy())

    if args.gradient_accumulation_steps > 1:
      loss = loss / args.gradient_accumulation_steps

    if fp16:
      with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    else:
      loss.backward()

    tr_loss += loss.item()
    if (step + 1) % args.gradient_accumulation_steps == 0:
      if fp16:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
      else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      model.zero_grad()
      global_step += 1
    if step > 0:
      epoch_iterator.set_description("Loss: %s" % (tr_loss/step) + " Acc: %s " % (correct/((step+1)*args.batch_size)))
    if (step + 1) % args.save_steps == 0:
      print('saving model to %s' % args.save_dir)
      model.save_pretrained(args.save_dir)
      tokenizer.save_pretrained(args.save_dir)
  print('saving model to %s' % args.save_dir)
  model.save_pretrained(args.save_dir)
  tokenizer.save_pretrained(args.save_dir)


def batch_input(all_features):
  input_ids_batch = []
  attention_mask_batch = []
  token_type_ids_batch = []
  i = 0
  for input_ids, attention_mask, token_type_ids in all_features:
    input_ids_batch.append(input_ids)
    attention_mask_batch.append(attention_mask)
    token_type_ids_batch.append(token_type_ids)
    i += 1
    if i % args.batch_size == 0:
      yield input_ids_batch, attention_mask_batch, token_type_ids_batch
      input_ids_batch = []
      attention_mask_batch = []
      token_type_ids_batch = []
  if len(input_ids_batch) > 0:
    yield input_ids_batch, attention_mask_batch, token_type_ids_batch


def eval(model):
  # load_and_cache_eval()
  qrels = []

  with open('./qrels.dev.small.tsv', 'r') as qrels_file:
    for line in tqdm(qrels_file, desc="loading qrels"):
      qid, _, cid, _ = line.rstrip().split('\t')
      qrels.append((qid, cid))

  dev_set = defaultdict(list)
  with open('./top1000.dev', 'r') as dev_file:
    for line in tqdm(dev_file, desc='loading dev file'):
      qid, cid, query, candidate = line.rstrip().split('\t')
      label = 1 if (qid, cid) in qrels else 0
      dev_set[query].append((candidate, label, qid))
  i = 0
  total_mrr = 0
  eval_iterator = tqdm(dev_set.items(), desc="Evaluating")
  for (query, choices) in eval_iterator:
    candidates = [choice[0] for choice in choices]
    labels = [choice[1] for choice in choices]
    if sum(labels) == 0: continue
    i += 1
    ranks = model.rank(query, candidates)
    total_mrr += 1/(np.sum(np.array(labels) * ranks) + 1)
    eval_iterator.set_description("Current rank: %s" % ranks[np.argmax(labels)] +
                                  " MRR: %s" % (total_mrr / i) + "Total: %s " % len(choices))


def main():
  from transformers_model import TransformersModel
  from bert_model import BertModel
  device, model, tokenizer = load_pretrained()
  if args.train:
    train(device, model, tokenizer)
  if args.eval:
    if args.model_class == 'bert_model':
      model = BertModel()
    else:
      model = TransformersModel()
    eval(model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--triples_path', default='triples.train.small.tsv')
  parser.add_argument('--steps', default=100000, type=int)
  parser.add_argument('--eval_steps', default=1000, type=int)
  parser.add_argument('--warmup', default=0.1, type=float)
  parser.add_argument('--save_steps', default=1000, type=int)
  parser.add_argument('--model', default='bert-base-uncased')
  parser.add_argument('--batch_size', default=8, type=int)
  parser.add_argument('--max_length', default=128, type=int)
  parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
  parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float,
                      help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
                      help="Max gradient norm.")
  parser.add_argument("--model_class", default='bert_model')
  parser.add_argument("--save_dir", default='./msmarco')
  parser.add_argument("--train", default=False, type=bool)
  parser.add_argument("--eval", default=False, type=bool)
  args = parser.parse_args()
  main()