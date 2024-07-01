import string
import pandas as pd
import numpy as np


# Loading data
df = pd.read_csv('dataset-a.csv', on_bad_lines='skip', delimiter="|",)
print('Dataset:\n', df)

# function replace symbol
def replace_symbols(text, replace_dict):
    for symbol, replacement in replace_dict.items():
        text = text.replace(symbol, replacement)
    return text

replace_dict = {
    '@': '',
    '#': '',
    '&': 'dan',
    '!': '',
    '"': '',
    "'": '',
    ",": '',
    '.': '',
    '?': '',
    '}\n\n': '',
    '\n': ' ',
    '\n\n': ''
}

# df['question'] = df['question'].apply(lambda x: replace_symbols(str(x), replace_dict))
# df['answer'] = df['answer'].apply(lambda x: replace_symbols(str(x), replace_dict))

def string_replace(original_string, to_replace, replacement):
    """
    Replace occurrences of 'to_replace' in 'original_string' with 'replacement'.

    Parameters:
    original_string (str): The string in which replacements will be made.
    to_replace (str): The substring to replace.
    replacement (str): The substring to replace with.

    Returns:
    str: The modified string with replacements made.
    """
    return original_string.replace(to_replace, replacement)

column_name = 'answer'

df['answer'] = df['answer'].apply(lambda x: string_replace(x, '\n', ''))

# Save the modified DataFrame to a new CSV file
df.to_csv('./dataset/modified_dataset-a.csv', index=False)

# Example to print the first few rows of the modified DataFrame
print(df.head())