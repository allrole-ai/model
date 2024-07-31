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
