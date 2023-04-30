"""
Implements the testing procedure and various cloze task metrics
"""

import torch
from transformers import BertForMaskedLM, BertTokenizer
from utils import load_test_data, load_test_target_words
 

def get_topk_predictions(model, tokenizer, k=5):
    """
    Get top k predictions for each masked word in between two verses.
    
    Args:
        model (BertForMaskedLM): pretrained model. 
        tokenizer (BertTokenizer): pretrained tokenizer. 
        k (int): determines how many top predictions (highest logits) are returned.
    
    Returns:
        top_k_preds (list): (# test sentences) * k. Each row stores the list of top k 
                            predictions for the masked word of that test line. 
    """

    # Load test data and get model output in logits
    test_dict = load_test_data()
    with torch.no_grad():
        masked_outputs_logits = model(**test_dict).logits

    # Get indices of masked tokens for each test line
    test_line_ids, masked_token_ids = torch.where(test_dict['input_ids'] == tokenizer.mask_token_id)
    
    # Initialize placeholders for the for-loop generating top k predictions  
    line_id, top_k_preds_words, topk_k_preds_ids, line_preds_ids, line_preds_tokens = 0, [], [], [None] * k, [None] * k

    for i in range(len(test_line_ids)): # For each masked token 
        token_id = masked_token_ids[i] # Get masked token id within that line (can have multiple)
        # Obtain the top k token ids
        preds_ids = torch.topk(masked_outputs_logits[line_id,token_id], k).indices
        # Convert the top k token ids into tokens
        preds_tokens = tokenizer.convert_ids_to_tokens(preds_ids)
        # Each target word might have gotten broken down into several tokens 
        # Put the token predictions (and their ids) for the same word together in a list 
        line_preds_ids = [old + [new] if old else [new] for old, new in zip(line_preds_ids, preds_ids)]
        line_preds_tokens = [old + [new] if old else [new] for old, new in zip(line_preds_tokens, preds_tokens)]
        
        if i == len(test_line_ids) - 1 or test_line_ids[i+1] > line_id:  # If the line id changes, save the tokens for the previous line
            line_id += 1                # and reinitialize the placeholders
            # Convert the tokens into a string (final prediction for the original word)
            topk_k_preds_ids.append(line_preds_ids)
            top_k_preds_words.append([tokenizer.convert_tokens_to_string(tokens) for tokens in line_preds_tokens])
            line_preds_ids, line_preds_tokens = [None] * k, [None] * k

    return topk_k_preds_ids, top_k_preds_words

def get_topk_accuracy(top_k_preds_words, targets):
    """
    Computes average top k accuracy of the model predictions on the test set.
    If any of the top k predictions matches the target word, score it as correct.

    Args:
        top_k_preds_words (list): 2nd output of `get_topk_predictions()`, an embedded list of words. 
        targets (list): list of target words (ground truths), output of `utils.load_test_target_words()`.

    Returns:
        accuracy (float): number of correct top k predictions / total number of targets to predict.
    """
    assert len(top_k_preds_words) == len(targets) # Sanity check
    accuracy_ct = 0
    for i in range(len(targets)):
        for j in range(len(top_k_preds_words[i])):
            if top_k_preds_words[i][j] == targets[i]:
                accuracy_ct += 1
                break
    
    return accuracy_ct / len(targets)

def get_topk_cossim(model, tokenizer, top_k_preds_ids, targets):
    """
    Computes average top k cosine similarity between the model predictions and the target words.
    top k cosine similarity = the highest cosine similarity among the top k predictions. 

    Args:
        model (BertForMaskedLM): pretrained model. 
        tokenizer (BertTokenizer): pretrained tokenizer. 
        top_k_preds_ids (list): 1st output of `get_topk_predictions()`, an embedded list of token ids. 
        targets (list): list of target words (ground truths), output of `utils.load_test_target_words()`.

    Returns:
        cossim (float): average top k cosine similarity computed across the test set.
    """
    assert len(top_k_preds_ids) == len(targets) # Sanity check

    # Get the BERT embedding layer to get word embedding vectors
    embedder = model.get_input_embeddings()
    # PyTorch implementation of cosine similarity between two vectors (both assumed to be 1D)
    cosine_similarity = torch.nn.CosineSimilarity(dim=0) 
    cossim_sum = 0 # Keep track of total cosine similarity score, to be divided by test set size
    with torch.no_grad():
        for i in range(len(targets)):
            # Tokenize the target word, convert list of tokens to tensor of ids, get embeddings
            label_emb = embedder( 
                torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(targets[i])), dtype=torch.int)
            )
            cossim = 0 # Keep track of highest cosine similarity among the top k predictions
            for j in range(len(top_k_preds_ids[i])): # Iterate through the top k predictions
                pred_emb = embedder(
                    torch.tensor(top_k_preds_ids[i][j], dtype=torch.int)
                )
                # Compute cosine similarity between current prediction and target
                temp_cossim = cosine_similarity(label_emb.reshape(-1), pred_emb.reshape(-1))
                if temp_cossim > cossim:
                    cossim = temp_cossim
            cossim_sum += cossim
    return cossim_sum / len(targets) # Return average cosine similarity
            

def get_topk_rhyme():
    # TODO
    raise NotImplementedError

def test_main(model_path='bert-base-uncased', k=5):
    """
    Main function to call when running the testing procedure. 

    Args:
        model_path (str): path to saved pretrained model. 
        k (int): metrics are computed over the top k predictions for each target. 

    Returns:
        results (dict): a dictionary of relevant average test metrics, 
                        including "accuracy", "cos_sim", and "rhyme".
    """
    # Fetch pretrained tokenizer (standard) and BERT model (fine-tuned by us)
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load predictions (in token ids and words) and targets
    topk_k_preds_ids, top_k_preds_words = get_topk_predictions(model, tokenizer, k)
    targets = load_test_target_words()

    acc = get_topk_accuracy(top_k_preds_words, targets)
    cossim = get_topk_cossim(model, tokenizer, topk_k_preds_ids, targets)
    #rhyme = get_topk_rhyme() #TODO

    return {"accuracy": acc,  "cos_sim": cossim}

print(test_main())
