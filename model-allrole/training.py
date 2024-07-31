import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

qa_data = pd.read_csv('dataset/qa-dataset.csv', delimiter='|')

qa_data = qa_data.dropna().reset_index(drop=True)

train_texts, val_texts, train_labels, val_labels = train_test_split(qa_data['question'], qa_data['answer'], test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(qa_data['answer'].unique()))

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

label2id = {label: i for i, label in enumerate(qa_data['answer'].unique())}
id2label = {i: label for label, i in label2id.items()}
