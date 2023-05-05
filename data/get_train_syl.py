"""
Merges each line of training data with corresponding syllabic information
Add '_' in between text and syllables to mark relative positions
"""

import os

TRAIN_FOLDER_PATH = "data/train_data_processed"
TRAIN_TYPE = "shakestrain" # MODIFY THIS, can be ['sonnets', 'shakestrain']

TEXT_PATH = os.path.join(TRAIN_FOLDER_PATH, f"{TRAIN_TYPE}_train_lines.txt")
SYL_PATH = os.path.join(TRAIN_FOLDER_PATH, f"{TRAIN_TYPE}_train_syllables.txt")

combined = []
with open(TEXT_PATH, 'r', encoding='utf-8') as texts, open(SYL_PATH, 'r', encoding='utf-8') as syls:
    for text, syl in zip(texts, syls):
        new_line = text.strip() + ' _ ' + syl.strip() # Underscore is a placeholder to locate position of the syllabic info
        combined.append(new_line)

# Save the lines for multimodal training (prev verse + next verse + corresponding syllabic info) 
with open(os.path.join(TRAIN_FOLDER_PATH, f"{TRAIN_TYPE}_train_textsyl.txt"), 'w') as f:
    for line in combined:
        f.write(f"{line}\n")