import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path='./trained_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(tokenizer, model, question, max_length=128, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    tokenizer, model = load_model_and_tokenizer()
    
    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ")
        if question.lower() == 'exit':
            break
        response = generate_response(tokenizer, model, question)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    main()
