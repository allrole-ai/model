import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd

def load_data(file_path, delimiter='|', encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, on_bad_lines='skip')
        print(df.head())  # Display the first few rows to check the format
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        # Display the first few lines of the file to help with debugging
        with open(file_path, 'r', encoding=encoding) as file:
            for i in range(5):
                print(file.readline())
        raise
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        # Display the first few lines of the file to help with debugging
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            for i in range(5):
                print(file.readline())
        raise
    
    # Check if the dataframe is loaded correctly
    print("Dataframe loaded successfully with shape:", df.shape)

    # Convert answers to numeric labels
    df['label'] = df['answer'].astype('category').cat.codes

    # Remove rows with invalid labels
    df = df[df['label'] >= 0]

    labels = tf.constant(df['label'].values)

    # Prepare the dataset
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
    input_ids = []
    attention_masks = []

    for question in df['question']:
        encoded = tokenizer.encode_plus(question, add_special_tokens=True, max_length=64, truncation=True, padding='max_length', return_attention_mask=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = tf.constant(input_ids)
    attention_masks = tf.constant(attention_masks)

    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_masks}, labels))
    dataset = dataset.shuffle(len(df)).batch(32)

    return dataset, len(df['label'].unique())

# Load the dataset
dataset, num_labels = load_data('dataset/qa.csv')

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=num_labels)

# Define custom training step function
@tf.function
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, outputs.logits, from_logits=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Compile and train the model
optimizer = tf.keras.optimizers.Adam()

for epoch in range(3):
    for batch in dataset:
        inputs, labels = batch
        loss = train_step(model, inputs, labels)
    print(f"Epoch {epoch + 1} completed")

# Save the model
model.save_pretrained('indobert_model')
