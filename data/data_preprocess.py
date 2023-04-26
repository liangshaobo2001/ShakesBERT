import json
import os

RAW_PATH = "data/raw_from_DeepSpeare"
PROCESSED_PATH = "data/masked_full_texts"
MODES = ["test", "train", "valid"]
END_SYMBOL = "<eos>"

for mode in MODES:
    masked_texts, full_texts, last_words = [], [], []
    for sonnet in open(os.path.join(RAW_PATH, f"sonnet_{mode}.txt"), 'r', encoding='utf-8'):
        # Placeholders for  1) Previous line with the last word masked
        #                   2) Previous line (full)
        #                   3) The masked word to predict (last word in the previous line)
        prev_masked, prev_full, prev_last_word = None, None, None

        for line in sonnet.strip().split(END_SYMBOL):
            stripped = line.strip() # Remove preceding and trailing spaces 
            if len(stripped) > 0: # If this line is not empty (artifact of split by `END_SYMBOL`) 
                words = stripped.split() # Get individual words in the line
                last_word = words[-1] # Extract the last word
                words[-1] = '[MASK]' # Replace the last word in the line with '[MASK]'
                masked = " ".join(words) 
                
                if prev_masked: # Concatenate previous line with current line
                    masked_text_line = prev_masked + ' ' + stripped
                    full_text_line = prev_full + ' ' + stripped
                    masked_texts.append(masked_text_line)
                    full_texts.append(full_text_line)
                    last_words.append(prev_last_word)

                prev_masked, prev_full, prev_last_word = masked, stripped, last_word
    
    # Save the output file
    masked_full_texts = {"masked_texts": masked_texts, "full_texts": full_texts, "last_words": last_words}
    out_file = open(os.path.join(PROCESSED_PATH, f"{mode}_masked_full.json"),"w")
    json.dump(masked_full_texts, out_file, indent=4)
    out_file.close()
            