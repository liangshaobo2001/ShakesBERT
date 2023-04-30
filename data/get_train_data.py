"""
Converts raw training data (from DeepSpeare) for transfer learning 
"""

import numpy as np
import os

DATA_FOLDER_PATH = "data"
RAW_FOLDER_PATH = "raw_from_DeepSpeare"
TRAIN_TYPE = "sonnets" # MODIFY THIS
PROCESSED_FOLDER_PATH = "train_data_processed"
END_SYMBOL = "<eos>"


# Placeholders to store full lines of sonnets for test = previous verse + next verse
full_texts = []

# For each sonnet (each line in the raw test file)
for sonnet in open(os.path.join(DATA_FOLDER_PATH, RAW_FOLDER_PATH, f"{TRAIN_TYPE}.txt"), 'r', encoding='utf-8'):
    # Placeholder to store previous verse   
    prev_full = None

    for line in sonnet.strip().split(END_SYMBOL): # For each verse in the current sonnet
        stripped = line.strip() # Remove preceding and trailing spaces 
        if len(stripped) > 0: # If this line is not empty (artifact of split by `end_symbol`)        
            
            if prev_full: # If this is not the first verse
                # Concat the previous verse with the current verse to form a test line
                full_text_line = prev_full + ' ' + stripped
                # Update the placeholder
                full_texts.append(full_text_line)

            prev_full = stripped

# Save the lines for training (prev verse + next verse) 
if not os.path.exists(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH)):
    os.mkdir(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH))
with open(os.path.join(DATA_FOLDER_PATH, PROCESSED_FOLDER_PATH, f"{TRAIN_TYPE}_train_lines.txt"), 'w') as f:
    for line in full_texts:
        f.write(f"{line}\n")