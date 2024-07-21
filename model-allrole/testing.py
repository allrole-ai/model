import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, AlbertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load tokenizer dan model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')

