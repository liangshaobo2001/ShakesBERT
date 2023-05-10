from flask import Flask, render_template, request
import torch, re
from transformers import BertForMaskedLM, BertTokenizer
from test import get_topk_predictions


model_path="final_model_shakestrain"
app = Flask(__name__)
target = "[MASK]"
model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()


@app.route('/', methods=['GET', 'POST'])
def index():
    return_sentence = None
    input_sentence = "There _ more life in one of your fair _ \n Than both your poets can in praise devise."
    input_number = 5
    input_text = "_"

    if request.method == 'POST':
        input_sentence = request.form.get('sentence')
        input_number = request.form.get('number')
        input_text = request.form.get('text')
        return_sentence = runmodel(input_sentence,input_number,input_text)
    return render_template('index.html', returned_sentence=return_sentence, input_sentence=input_sentence, input_number = input_number, input_text=input_text)

# def reverse_sentence(sentence):
#     words = sentence.split()
#     reversed_words = ' '.join(reversed(words))
#     return reversed_words

def predict_masked_tokens_bench(text, tokenizer, model, top_k=3):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits

    predictions = []
    for mask_position in mask_positions:
        probs = logits[0, mask_position].softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(probs, top_k)
        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

        mask_predictions = []
        for i in range(top_k):
            mask_predictions.append({
                "score": top_k_values[i].item(),
                "token": top_k_indices[i].item(),
                "token_str": top_k_tokens[i]
            })
        predictions.append(mask_predictions)

    return predictions

def runmodel(sentence,number,text):
    sentence = sentence.replace(text, " [MASK] ")
    predicted = predict_masked_tokens_bench(sentence, tokenizer, model, top_k=int(number))
    replaced_text = re.sub(r"\[MASK\]", "{}", sentence)
    replaced_text = replaced_text.format(*[[tok['token_str'] for tok in pred] for pred in predicted])
    return replaced_text
    

if __name__ == '__main__':
    app.run(threaded=True)
