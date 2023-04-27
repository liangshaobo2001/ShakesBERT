import os
import torch

TEST_FOLDER_PATH = "data/test_data_processed"

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
        labels = eval(f.read())
    
    return {'input_ids':input_ids, 
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids, 
            'labels':labels}

