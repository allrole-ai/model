

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Simpan model dan tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Langkah 4: Pengujian dan Evaluasi Model
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


test_questions = [
    "cara bayar spp",
    "cara buka Menu"
]

for question in test_questions:
    print(f"Question: {question}")
    print(f"Answer: {generate_response(question)}\n")
