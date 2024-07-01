from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import io
import os
import re
import requests
import csv
import datetime
import numpy as np
import pandas as pd
import random
import pickle


factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham","kurang ngerti","I don't know"]

# Loading data
df = pd.read_csv('dataset-a.csv', on_bad_lines='skip', delimiter="|",)
print('Dataset:\n', df)

# function replace symbol
def replace_symbols(text, replace_dict):
    for symbol, replacement in replace_dict.items():
        text = text.replace(symbol, replacement)
    return text

def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower())
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
  sentence = sentence.replace('teung', '')
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
  sentence = ' '.join(sentence.split())
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence)
  return sentence