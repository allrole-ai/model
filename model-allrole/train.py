import pandas as pd
import csv
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# load dataset train
df = pd.read_csv('lar-clean.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)

# Buat label biner (0 atau 1) dari data jawaban jika perlu
df['label'] = df.index % 2  # For example, using indexes as temporary labels

# Pisahkan data menjadi pelatihan dan pengujian
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# create hugging face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')

#preprocessing data
def preprocess_function(examples):
    inputs = tokenizer(
        examples['question'], 
        examples['answer'], 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )
    inputs['labels'] = examples['label']
    return inputs
