from datasets import Dataset, load_dataset
import numpy as np
import os
import torch
from transformers import BertTokenizer

TEST_FOLDER_PATH = "data/test_data_processed"
TRAIN_FOLDER_PATH = "data/train_data_processed"

def load_test_data():
    """
    Helper function to load test data into a dictionary
    Pass 'input_ids', 'attention_mask', and 'token_type_ids' to BERT
    Evaluate model output against 'labels' (ground truths)
    """
    input_ids = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_input_ids.pt'))
    attention_mask = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_attention_mask.pt'))
    token_type_ids = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_token_type_ids.pt'))
    with open(os.path.join(TEST_FOLDER_PATH, 'test_labels.txt'), 'r') as f:
        labels = f.read().splitlines()
    
    return {'input_ids':input_ids, 
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids, 
            'labels':labels}

def load_train_eval_data(train_type, train_prop=.75):
    """
    Helper function to load and split train/eval data as TensorDatasets

    In the datasets, 
        1st col = `input_ids` (to be masked by a data_collator)
        2nd col = `token_type_ids`
        3rd col = `attention_mask`
        4th col = `labels` (ground truths)
    
    Can be passed as values for transformers.Trainer(train_dataset=xxx, val_dataset=xxx).

    Args:
        train_type (str): Type of training (determines relevant data file name).
        train_prop (float): Proportion of training data out of the entire dataset.
                            Val data will take up (1-`train_prop`) of the original dataset. 
    Returns:
        train_set (Dataset): Training set in Hugging Face dataset format
        val_set (Dataset): Training set in Hugging Face dataset format
    """
    # Load processed "prev verse + next verse" texts
    full_texts = load_dataset("text", data_files=os.path.join(TRAIN_FOLDER_PATH, f"{train_type}_train_lines.txt"))

    # Tokenize the processed texts
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = full_texts.map(
        lambda x: tokenizer(x["text"], max_length=60, padding='max_length', truncation=True), 
        batched=True, 
        remove_columns=["text"]
    )

    # Duplicate the `input_ids` column (will be masked) to create `labels` (ground truths)
    def add_labels(x):
        x['labels'] = x['input_ids'].copy()
        return x
    labeled_texts = tokenized_texts.map(add_labels, batched=True)

    # Make the dataset compatible with torch formats (i.e., convert to tensors)
    torch_texts = labeled_texts.with_format("torch")

    # Do train-val split and return
    split_texts = torch_texts['train'].train_test_split(train_size=train_prop)
    return split_texts['train'], split_texts['test']

