

def load_model_and_tokenizer(model_path='./trained_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(tokenizer, model, question, max_length=128):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
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
