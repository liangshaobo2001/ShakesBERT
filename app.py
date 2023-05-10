from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    reversed_sentence = None
    if request.method == 'POST':
        sentence = request.form['sentence']
        reversed_sentence = reverse_sentence(sentence)
    return render_template('index.html', reversed_sentence=reversed_sentence)

def reverse_sentence(sentence):
    words = sentence.split()
    reversed_words = ' '.join(reversed(words))
    return reversed_words

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
