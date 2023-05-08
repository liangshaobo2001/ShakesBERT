from datasets import load_dataset
import os
import torch
from transformers import BertTokenizer, DataCollatorForLanguageModeling

TEST_FOLDER_PATH = "data/test_data_processed"
TRAIN_FOLDER_PATH = "data/train_data_processed"

def load_test_data():
    """
    Helper function to load test data into a dictionary
    Pass **output_dict to BERT model
    """
    input_ids = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_input_ids.pt'))
    attention_mask = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_attention_mask.pt'))
    token_type_ids = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_token_type_ids.pt'))
    # labels = torch.load(os.path.join(TEST_FOLDER_PATH, 'test_labels.pt'))
    
    return {
        'input_ids':input_ids, 
        'attention_mask': attention_mask, 
        'token_type_ids': token_type_ids, 
        # 'labels': labels
    }

def load_test_target_words():
    """ Helper function to load ground truth target words for testing """
    with open(os.path.join(TEST_FOLDER_PATH, 'test_last_words.txt'), 'r') as f:
        target_words = f.read().splitlines()
    return target_words

def load_test_masked_verses():
    """ Helper function to load test lines with target word masked """
    with open(os.path.join(TEST_FOLDER_PATH, 'test_masked_verses.txt'), 'r') as f:
        masked_verses = f.read().splitlines()
    return masked_verses

def load_train_eval_data(train_type, multimodal=False, train_prop=.75):
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
    if multimodal:
        # Load processed "prev verse + next verse" texts & corresponding syllabic info
        full_texts = load_dataset("text", data_files=os.path.join(TRAIN_FOLDER_PATH, f"{train_type}_train_textsyl.txt"))
    else: 
        # Load processed "prev verse + next verse" texts
        full_texts = load_dataset("text", data_files=os.path.join(TRAIN_FOLDER_PATH, f"{train_type}_train_lines.txt"))

    # Tokenize the processed texts
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = full_texts.map(
        lambda x: tokenizer(x["text"], max_length=120, padding='max_length', truncation=True), 
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

def get_syl_mask(tokenizer, inputs):
    """
    Returns a mask indicating locations of syllabic information

    Args:
        inputs (tensor): token ids, dimension = batch_size x 120 (max length w/ padding). 
    
    Returns:
        syl_mask (tensor):  boolean masks, TRUE if syllabic info at current location in inputs,
                            same dimension as inputs. 
    """
    
    under_id = tokenizer.convert_tokens_to_ids('_') # BERT token id for underscore = 1035
    _, loc = (inputs == under_id).nonzero(as_tuple=True) # Get list of indices where the underscore appears

    # Placeholders to store mask for syllabic info, dim = batch_size x 120
    syl_mask = torch.full(size=inputs.shape, fill_value=0)

    for i in range(len(inputs)): # For each line of input 
        syl_mask[i, loc[i]:] = torch.tensor([1]*len(syl_mask[i, loc[i]:])) # Mark non-text tokens as 1

    return syl_mask.to(torch.bool)


class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator to load multi-modal training data without masking syllabic info
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Edit the original `torch_mask_tokens()` implementation to avoid masking syllabic info
        """
        # Get a mask indicating locations of the syllabic information
        # Also, remove the '_' in the training data (only there to signal the position of syllabic info)
        syl_mask = get_syl_mask(self.tokenizer, inputs)
        
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Get a mask indicating locations of the special tokens (e.g., [CLS], [SEP]) 
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Make sure that the special tokens and tokens corresponding to syllabic info are not masked
        probability_matrix.masked_fill_(special_tokens_mask | syl_mask, value=0.0) # NEW
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels