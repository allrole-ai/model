import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path='./trained_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(tokenizer, model, question, max_length=128):
    inputs = tokenizer.encode(question, return_tensors='pt')
