import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, AlbertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load tokenizer dan model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')

# Load Encoder label
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

