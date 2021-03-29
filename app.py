from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import pandas as pd
import sklearn
import pickle
import os

ml = 'classificate3.pkl'
clf = pickle.load(open(ml, 'rb'))
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.config.update(dict(
    SECRET_KEY="powerfull key",
    WTF_CSRF_SECRET_KEY="mmm a csrf secret key"
))

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    dataset = FileField()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=('GET', 'POST'))
def home():
    form = MyForm()
    if form.validate_on_submit():
        filename = form.dataset.data
        df = pd.read_csv(filename)
        names = df[form.name.data]
        predicted = clf.predict(names)
        result = pd.DataFrame({'Наименование': names, 'Категория': predicted})
        result.to_csv('predicted.csv', index=False)
        df1 = pd.read_csv('predicted.csv')
        
        return render_template('result1.html', prediction=df1['Категория'].head(5), name=df1['Наименование'].head(5))

    return render_template('home.html', form=form)


@app.route('/uploadjs', methods=['GET', 'POST'])
def uploadjs_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            name = ('predicted1.csv')
            os.rename(filename, name)
            return 'file uploaded'

@app.route('/uppredjs', methods=['GET'])
def uppredjs():
    df = pd.read_csv('predicted1.csv')
    names = df['Наименование']
    df5 = df[names]
    predicted = clf.predict(df5)
    result = pd.DataFrame({'Наименование': names, 'Категория': predicted})
    result.to_csv('predicted1.csv', index=False)
    return 


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


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=( ", ".join( repr(e) for e in my_prediction ) ), name=message)


@app.route('/predictjs', methods=['GET','POST'])
def predictjson():
    if request.method == 'POST':
        content = request.get_json()
        param = content['text'].split(',')
        param =[str(text) for text in param]
        my_prediction1 = clf.predict(param)
        pred = {'Название': str(param),
        'Категория': str(my_prediction1)}
    return jsonify(pred)


if __name__ == '__main__':
    app.run(debug=False)
