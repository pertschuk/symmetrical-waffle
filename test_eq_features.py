
def main():
  with open('tf_features.txt') as tf_features, open('pt_features.txt') as pt_features:
    for tf_line, pt_line in zip(tf_features, pt_features):
      try:
        assert tf_line == pt_line
      except:
        print('***TF LINE***')
        print(tf_line)
        print('***PT LINE***')
        print(pt_line)

if __name__ == '__main__':
  main()