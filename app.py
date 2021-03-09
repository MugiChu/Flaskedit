from flask import Flask, render_template, request
import pandas as pd
import sklearn
import pickle
import string
import re
import nltk
nltk.download('stopwords')
from nltk.stem import *
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
nltk.download('punkt')
from nltk import word_tokenize

mystem = Mystem()

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['…', '«', '»', '...'])


def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
    return " ".join(tokens)


def lemmatize_text(text):
    text_lem = mystem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ']
    return " ".join(tokens)


def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])


def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)


def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])


filename1 = 'lemm.pkl'
lemm = pickle.load(open(filename1, 'rb'))

filename2 = 'stop_words.pkl'
sp = pickle.load(open(filename2, 'rb'))

filename3 = 'remove_spaces.pkl'
rs = pickle.load(open(filename3, 'rb'))

filename4 = 'remove_numbers.pkl'
rn = pickle.load(open(filename4, 'rb'))

filename5 = 'r_p.pkl'
rp = pickle.load(open(filename5, 'rb'))

filename = 'classificate.pkl'
clf = pickle.load(open(filename, 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data1 = rp(data).lower()
        data2 = rn(data1)
        data3 = rs(data2)
        data4 = sp(data3)
        data5 = lemm(data4)
        my_prediction = clf.predict(data5)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=False)
