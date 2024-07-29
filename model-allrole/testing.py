import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path='./trained_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_response(tokenizer, model, question, max_length=128):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits
preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)
test_accuracy_metric.update_state(labels_test, preds_test)

test_accuracy = test_accuracy_metric.result()
print(f"Test accuracy: {test_accuracy:.4f}")

# Loop interaktif untuk input & prediksi pengguna
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = predict(question)
    print(f"Predicted Answer: {answer}")