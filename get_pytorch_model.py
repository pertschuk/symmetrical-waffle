from transformers import *
import os


def main():
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='./')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./')
  save_dir = './pt_model'
  os.makedirs(save_dir)
  model.save_pretrained(save_dir)
  tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
  main()