import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

df = pd.read_csv('dataset/qa-dataset.csv', delimiter='|', on_bad_lines='skip')

df.columns = ['question', 'answer']

def filter_valid_rows(row):
    return len(row) == 2

print("Nama kolom dalam DataFrame:", df.columns)
print("Beberapa baris data:")
print(df.head())

df['question'] = df['question'].astype(str)

df['answer'] = df['answer'].astype(str)

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

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

dataset = CustomDataset(df['question'].tolist(), df['answer'].tolist(), tokenizer)

train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

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

