import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

df = pd.read_csv('dataset/qa-dataset.csv', delimiter='|', on_bad_lines='skip')

df.columns = ['question', 'answer']
