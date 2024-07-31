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

