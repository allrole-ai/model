import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


    input_ids_test.append(encoded['input_ids'])
    attention_masks_test.append(encoded['attention_mask'])

input_ids_test = tf.concat(input_ids_test, axis=0)
attention_masks_test = tf.concat(attention_masks_test, axis=0)
labels_test = tf.constant(df_test['encoded_answer'].values)

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