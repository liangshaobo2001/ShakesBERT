"""
Implements the testing procedure and various cloze task metrics
"""

import torch
from transformers import BertForMaskedLM, BertTokenizer
from utils import load_test_data
 

def get_topk_predictions(model_path='bert-base-uncased', k=5):
    """
    Get top k predictions for each masked word in between two verses.
    
    Args:
        model_path (str): where the pretrained BertForMaskedLM is saved.
        k (int): determines how many top predictions (highest logits) are returned.
    
    Returns:
        top_k_preds (list): (# test sentences) * k. Each row stores the list of top k 
                            predictions for the masked word of that test line. 
    """
    # Fetch pretrained tokenizer (standard) and BERT model (fine-tuned by us)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained(model_path)

    # Load test data and get model output in logits
    test_dict = load_test_data()
    with torch.no_grad():
        masked_outputs_logits = model(**test_dict).logits

    # Get indices of masked tokens for each test line
    test_line_ids, masked_token_ids = torch.where(test_dict['input_ids'] == tokenizer.mask_token_id)
    
    # Initialize placeholders for the for-loop generating top k predictions  
    line_id, top_k_preds, line_preds = 0, [], [None] * k

    for i in range(len(test_line_ids)): # For each masked token 
        if test_line_ids[i] > line_id:  # If the line id changes, save the tokens for the previous line
            line_id += 1                # and reinitialize the placeholders
            # Convert the tokens into a string (final prediction for the original word)
            top_k_preds.append([tokenizer.convert_tokens_to_string(tokens) for tokens in line_preds])
            line_preds = [None] * k
        token_id = masked_token_ids[i] # Get masked token id within that line (can have multiple)
        # Convert the top k token ids into tokens
        preds = tokenizer.convert_ids_to_tokens(torch.topk(masked_outputs_logits[line_id,token_id], k).indices)
        # Each target word might have gotten broken down into several tokens 
        # Put the token predictions for the same word together in a list 
        line_preds = [old + [new] if old else [new] for old, new in zip(line_preds, preds)]
    
    return top_k_preds

def get_topk_accuracy(k=5):
    # TODO
    raise NotImplementedError

def get_topk_cossim(k=5):
    # TODO
    raise NotImplementedError

def get_topk_rhyme(k=5):
    # TODO
    raise NotImplementedError

        
        

