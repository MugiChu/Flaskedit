from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sklearn
import pickle
import os

filename = 'classificate1.pkl'
clf = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

app.config.update(dict(
    SECRET_KEY="powerfull key",
    WTF_CSRF_SECRET_KEY="mmm a csrf secret key"
))

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    dataset = FileField()

@app.route('/', methods=('GET', 'POST'))
def home():
    form = MyForm()
    if form.validate_on_submit():
        f = form.dataset.data
        df = pd.read_csv(f)
        names = df[form.name.data]
        predicted = clf.predict(names)
        result = pd.DataFrame({'Наименование': names, 'Категория': predicted})
        result.to_csv('predicted.csv', index=False)
        df1 = pd.read_csv('predicted.csv')
        
        return render_template('result1.html', prediction=df1['Категория'].head(5), name=df1['Наименование'].head(5))


    return render_template('home.html', form=form)


@app.route('/result1')
def send():
    try:
        df3 = 'predicted.csv'
        return send_file(
            df3,
            mimetype='text/csv',
            attachment_filename=df3,
            as_attachment=True)
    except Exception as e:
        return str(e)


@app.route('/avg/<nums>')
def avg(nums):
    return 'User %s' % nums


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=( ", ".join( repr(e) for e in my_prediction ) ), name=message)

@app.route('/predictjs', methods=['POST'])
def predictjson():
    if request.method == 'POST':
        content = request.get_json()
        param = content['text'].split(',')
        param =[str(text) for text in param]
        my_prediction1 = clf.predict(param)
        pred = {'category': str(my_prediction1)}
    return jsonify(pred)


if __name__ == '__main__':
    app.run(debug=False)
