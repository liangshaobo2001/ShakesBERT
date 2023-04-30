"""
Converts raw test data (from DeepSpeare) for verse-final word prediction task 
"""

import numpy as np
import os
from transformers import BertTokenizer
import torch

DATA_FOLDER_PATH = "data"
RAW_FOLDER_PATH = "raw_from_DeepSpeare"
RAW_TEST_FILE = "temp.txt" # MODIFY THIS
PROCESSED_FOLDER_PATH = "test_data_processed"
END_SYMBOL = "<eos>"

# Get pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Placeholders to store
#   1) Full lines of sonnets for test = previous verse + next verse
#   2) Start index and end index (+1) for the tokens associated with the test word
#   3) The test words (answers) 
full_texts, start_end_ids, last_words = [], [], []

# For each sonnet (each line in the raw test file)
for sonnet in open(os.path.join(DATA_FOLDER_PATH, RAW_FOLDER_PATH, RAW_TEST_FILE), 'r', encoding='utf-8'):
    # Placeholders to store
    #   1) Previous verse   
    #   2) The last word in the previous verse
    #   3/4) The start and end (+1) index of the tokens associated with the last word
    prev_full, prev_last_word, prev_start_id, prev_end_id = None, None, None, None

    for line in sonnet.strip().split(END_SYMBOL): # For each verse in the current sonnet
        stripped = line.strip() # Remove preceding and trailing spaces 
        if len(stripped) > 0: # If this line is not empty (artifact of split by `end_symbol`) 
            words = stripped.split() # Get individual words in the line
            last_word = words[-1] # Extract the last word
            words[-1] = '_' # Insert a special symbol for start index identification
            words.append(last_word)
            marked = " ".join(words)
            start_id = tokenizer.tokenize(marked).index('_') + 1 # Get start index (+1 for [CLS])
            tokenized_text = tokenizer.tokenize(stripped)
            end_id = len(tokenized_text) + 1          
            
            if prev_full: # If this is not the first verse
                # Concat the previous verse with the current verse to form a test line
                full_text_line = prev_full + ' ' + stripped

                # Update the placeholders
                full_texts.append(full_text_line)
                start_end_ids.append([prev_start_id, prev_end_id])
                last_words.append(prev_last_word)

            prev_full, prev_last_word, prev_start_id, prev_end_id = stripped, last_word, start_id, end_id

# Get the maximum number of tokens in each verse
# Double that for the max length passed to the tokenizer
max_len = np.max([l[1] for l in start_end_ids])

# Tokenize the test lines
inputs = tokenizer(full_texts, max_length=max_len * 2, padding='max_length')

# # Save the ground truth labels (copy of inputs['input_ids'], not masked yet)
# torch.save(torch.tensor(inputs['input_ids'], dtype=torch.int), 
#            os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, "test_labels.pt"))

# Replace test tokens with [MASK] (id=103)
for i in range(len(inputs['input_ids'])):
    start_id, end_id = start_end_ids[i][0], start_end_ids[i][1]
    inputs['input_ids'][i][start_id:end_id] = [tokenizer.mask_token_id] * (end_id-start_id)

# Save the individual tensors for testing
for key in inputs.keys():
    inputs[key] = torch.tensor(inputs[key],dtype=torch.int)
    torch.save(inputs[key], os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, f"test_{key}.pt"))

# Save the list of test words (ground truths)
if not os.path.exists(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH)):
    os.mkdir(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH))
with open(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, "test_last_words.txt"), 'w') as f:     
    for word in last_words:
        f.write(f"{word}\n")
