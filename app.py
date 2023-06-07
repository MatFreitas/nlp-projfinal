from flask import Flask, render_template, request
import joblib
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
# from keras.utils import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from transformers import AutoTokenizer, BloomForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

# tokenizer = Tokenizer()
detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)

# def generate_text(seed_text, next_words, model, max_sequence_len):
#     for _ in range(next_words):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#         predicted = model.predict(token_list, verbose=0)
        
#         output_word = ""
#         for word,index in tokenizer.word_index.items():
#             if index == predicted.all():
#                 output_word = word
#                 break
#         seed_text += " "+output_word
#     return seed_text.title()

nltk_model = joblib.load('nltk_model.joblib')
# bloom_model = joblib.load('bloom_model.joblib')
# lstm_model = joblib.load('lstm_model.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    print(request.method)
    if request.method == 'POST':
        # LSTM
        # lstm_response = generate_text("united states", 5, lstm_model, 10000)

        if request.form.get('NLTK') == 'NLTK':
            ntlk_response = generate_sent(nltk_model, 20, random_seed=random.randint(5, 200))
            return render_template("index.html", nltk_response=ntlk_response)
    
        
        else:
            # pass # unknown
            return render_template("index.html")
    elif request.method == 'GET':
        # return render_template("index.html")
        print("No Post Back Call")
    return render_template("index.html")
