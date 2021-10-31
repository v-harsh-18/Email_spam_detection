from flask import Flask,render_template,request,session,redirect,flash
import os
from textblob import TextBlob
import numpy as np
import pandas as pd
import nltk
from werkzeug.utils import secure_filename, send_file
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/email', methods=['POST'])
def emotion():

    csvf = request.files['csv']
    csvname = secure_filename(csvf.filename)
    csvf.save(os.path.join(app.config['UPLOAD_FOLDER'], csvname))

    opencsv='static/files/'+csvname

    dataframe = pd.read_csv(opencsv)

    dataframe.drop_duplicates(inplace = True)

    dataframe.isnull().sum()

    nltk.download('stopwords')

    def process_text(text):

     nopunc = [char for char in text if char not in string.punctuation ]
     nopunc = ''.join(nopunc)

     clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

     return clean_words

    dataframe['EmailText'].head().apply(process_text)

    from sklearn.feature_extraction.text import CountVectorizer
    messages_bow = CountVectorizer(analyzer = process_text).fit_transform(dataframe['EmailText'])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(messages_bow, dataframe['Label'], test_size=0.20, random_state=42)

    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB().fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    pred = classifier.predict(X_train)
    print('Accuracy: ', round(accuracy_score(y_train, pred),4))

    result=100*(round(accuracy_score(y_train, pred),4))

    return render_template('result.html',result=result)
 
app.run(debug=True)
