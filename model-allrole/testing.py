




def generate_response(tokenizer, model, question, max_length=128):
    inputs = tokenizer.encode(question, return_tensors='pt')


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
