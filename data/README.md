# Data loading

## Folders
The `data/` folder houses all training and test data and their preprocessing scripts. (The Google drive data folders are structured in the exact same way.) There are three folders in the `data/` folder:

1. `raw_from_DeepSpeare/`: raw poetry files downloaded from the DeepSpeare project. The files follow the following format: every line is a complete poem, and a `<eos>` tag separates two verses. 
2. `test_data_processed/`: The test files are obtained from their raw forms stored in `data/raw_from_DeepSpeare/shakestest.txt` using the `data/get_test_data.py` script. The files include outputs of `BertTokenizer`, including the tensors storing attention masks, input token ids, and token type ids, respectively, and the ground truth target words. 
3. `train_data_processed/`: Each file is used for its own round of fine-tuning. Every line in each training file contains two neighboring verses in a given poem. 

## Load training data

```
from utils import load_train_eval_data

# Among the original training data, 75% is used for training, 25% is used for validation
train_prop = 0.75

# Set the fine-tuning type 
train_type = "sonnets" # One of ["sonnets", "shakestrain", "poems"]

train_set, val_set = load_train_eval_data(train_type, train_prop)

# PROFIT?
```

The `utils.load_train_eval_data` handles the training data loading process and returns a tuple of two Hugging Face Datasets `(train_set, val_set)`. 

Do specify the type of training data to load. There are three possible training types:
1. `"poems"`: Note that `poems_train_lines.txt` is stored in the Google Drive folder and contains all non-sonnet poems from Gutenberg. 
2. `"sonnets"`: All sonnets from Gutenberg that are not written by Shakespeare. 
3. `"shakestrain"`: (~80% of) all sonnets written by Shakespeare (n=123). 

For an example of using Hugging Face datasets during the training process, see this [Hugging Face tutorial](https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt#fine-tuning-distilbert-with-the-trainer-api). 

## The testing procedure

```
from test import test_main

model_path = "I/stored/my/fine-tuned/model/here"
k = 5 # I want top 5 test metrics

results_dict = test_main(model_path, k)

print(f"Top {k} accuracy is {results_dict["accuracy"]}.")
print(f"Top {k} cosine similarity score is {results_dict["cossim"]}.")
# print(f"Top {k} rhyming score is {results_dict["rhyme"]}.") # To be implemented
```

