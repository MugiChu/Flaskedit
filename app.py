from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sklearn
import pickle
import os

filename = 'classificate.pkl'
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
        #filename = form.name.data +'.csv'
        #f.save(os.path.join(
        #   filename
        #))

        df = pd.read_csv(f)
        names = df[form.name.data]
        predicted = clf.predict(names)
        result = pd.DataFrame({'Наименование': names, 'Категория': predicted})
        result.to_csv('predicted.csv', index=False)
        df1 = pd.read_csv('predicted.csv')


        return render_template('result1.html', prediction=df1['Категория'].head(5), name=df1['Наименование'].head(5))
    return render_template('home.html', form=form)


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


if __name__ == '__main__':
    app.run(debug=False)
