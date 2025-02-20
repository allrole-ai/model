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
    
train_dataset = QADataset(train_encodings, train_labels)

val_dataset = QADataset(val_encodings, val_labels)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch"  # Updated parameter name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Melatih model
trainer.train()

# Evaluasi model
trainer.evaluate()

# Simpan model dan tokenizer
if not os.path.exists('model'):
    os.makedirs('model')
    
model.save_pretrained('model/qa_model')

tokenizer.save_pretrained('model/qa_tokenizer')

# Fungsi untuk menjawab pertanyaan baru
nlp = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def answer_question(question, context=[]):
    """
    Menjawab pertanyaan berdasarkan konteks percakapan sebelumnya.
    
    Parameters:
    - question (str): Pertanyaan yang diajukan.
    - context (list): Daftar konteks percakapan sebelumnya.
    
    Returns:
    - str: Jawaban yang dihasilkan oleh model.
    """
    context.append(question)
    input_text = " ".join(context)
    result = nlp(input_text)
    label = max(result[0], key=lambda x: x['score'])['label']
    return id2label[int(label.split('_')[-1])]

if __name__ == "__main__":
    context = []
    question = "What is AI?"
    answer = answer_question(question, context)
    print("Answer:", answer)





