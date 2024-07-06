import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('indobert_model')
model = TFBertForSequenceClassification.from_pretrained('indobert_model')

# Load the label encoder
with open('qa.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])
