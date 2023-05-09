"""
Demo script for one-word prediction. 
"""

import argparse
from test import get_topk_predictions
import torch
from transformers import BertForMaskedLM, BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--masked_input', '-mi', help="Input verses with target word masked with _ ")
parser.add_argument('--target', '-t', help="The masked target word (ground truth)")
parser.add_argument('--model_path', '-mp', help="Directory of pretrained model", default='bert-base-uncased')
parser.add_argument('--k', '-k', help="Will return top k predictions", type=int, default=5)

args = parser.parse_args()

def prep_test_format(tokenizer, masked_input, target):
    """
    Similar function as `get_test_data.py` except for individual sentences via command line input.
    Prepare masked and tokenized input for the model to generate predictions.

    Args:
        tokenizer (BertTokenizer): pretrained tokenizer. 
        masked_input (str): input verses with the target word masked with ONE underscore 
        target (str): target word, ground truth. 
    
    Returns:
        test_dict (dict): Dictionary of BERT tokenizer output. 
    """
    start_id = tokenizer.tokenize(masked_input).index('_') + 1
    len_target_token = len(tokenizer.tokenize(target))
    end_id = start_id + len_target_token
    
    full_text = masked_input.replace('_', target)
    input = tokenizer(full_text)
    input['input_ids'][start_id:end_id] = [tokenizer.mask_token_id] * (end_id-start_id)

    for key in input.keys():
        input[key] = torch.tensor(input[key],dtype=torch.int).reshape(1,-1)

    return input


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained(args.model_path)

    test_dict = prep_test_format(tokenizer, args.masked_input.strip(), args.target)
    _, top_k_preds_words = get_topk_predictions(model, tokenizer, test_dict, args.k)

    print()
    print("=========DEMO RESULTS=========")
    print(f"For the input verses: \'{args.masked_input}\',")
    print(f"The target word is \'{args.target}\'.")
    print(f"The top {args.k} predictions are: {top_k_preds_words[0]}.")

