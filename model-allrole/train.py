import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset


# Langkah 1: Persiapan Data
df = pd.read_csv('dataset/qa-dataset.csv', delimiter='|', on_bad_lines='skip')

# Ganti nama kolom jika perlu
df.columns = ['question', 'answer']


# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

# Verifikasi kolom
print("Nama kolom dalam DataFrame:", df.columns)
print("Beberapa baris data:")
print(df.head())

# Pastikan kolom 'question' dan 'answer' adalah string
df['question'] = df['question'].astype(str)
df['answer'] = df['answer'].astype(str)

# Langkah 2: Tokenisasi dan Pembuatan Model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Menambahkan token padding ke tokenizer
tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        outputs = self.tokenizer(answer, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)

        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels = outputs['input_ids'].squeeze(0)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Buat dataset
dataset = CustomDataset(df['question'].tolist(), df['answer'].tolist(), tokenizer)

# Bagi dataset menjadi train dan eval
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])


# Langkah 3: Pelatihan Model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Masked Language Modeling is not used for causal language models
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()



# Langkah 4: Pengujian dan Evaluasi Model
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




for question in test_questions:
    print(f"Question: {question}")
    print(f"Answer: {generate_response(question)}\n")

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, pipeline

# Membaca dataset QA
qa_data = pd.read_csv('dataset/qa-dataset.csv', delimiter='|')

# Pra-pemrosesan data
qa_data = qa_data.dropna().reset_index(drop=True)

# Split data menjadi train dan test
train_texts, val_texts, train_labels, val_labels = train_test_split(qa_data['question'], qa_data['answer'], test_size=0.2, random_state=42)

# Load tokenizer dan model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(qa_data['answer'].unique()))

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
    output_dir='./results_bert',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_bert',
    logging_steps=10,
    eval_strategy="epoch"
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
if not os.path.exists('model_bert'):
    os.makedirs('model_bert')

model.save_pretrained('model_bert/qa_model')
tokenizer.save_pretrained('model_bert/qa_tokenizer')

# Fungsi untuk menjawab pertanyaan baru
nlp = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def answer_question(question, context=[]):
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
