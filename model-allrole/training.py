import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

# Membaca dataset QA
qa_data = pd.read_csv('dataset/qa-dataset.csv', delimiter='|')

# Pra-pemrosesan data
qa_data = qa_data.dropna().reset_index(drop=True)

# Split data menjadi train dan test
train_texts, val_texts, train_labels, val_labels = train_test_split(qa_data['question'], qa_data['answer'], test_size=0.2, random_state=42)

# Load tokenizer dan model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(qa_data['answer'].unique()))

# Tokenisasi
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Konversi label menjadi angka
label2id = {label: i for i, label in enumerate(qa_data['answer'].unique())}
id2label = {i: label for label, i in label2id.items()}
train_labels = train_labels.map(label2id)
val_labels = val_labels.map(label2id)

# Membuat dataset dengan format yang sesuai untuk Trainer
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)







