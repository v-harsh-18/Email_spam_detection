from flask import Flask,render_template,request,session,redirect,flash
import math
from textblob import TextBlob
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app=Flask(__name__)

#Read the csv file
dataframe = pd.read_csv('spam.csv')

# Remove duplicates
dataframe.drop_duplicates(inplace = True)

# Check the number of missing (NA) data for each column
dataframe.isnull().sum()
# print(dataframe.isnull().sum())

#Download the stopwords package
nltk.download('stopwords')

def process_text(text):

    #remove punctuation
    nopunc = [char for char in text if char not in string.punctuation ]
    nopunc = ''.join(nopunc)

    # remove stopwords
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    #return a list of clean words
    return clean_words

dataframe['EmailText'].head().apply(process_text)

# Convert a collection of text to a matrix of tokens(tokenization)
messages_bow = CountVectorizer(analyzer = process_text).fit_transform(dataframe['EmailText'])

# Spliting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(messages_bow, dataframe['Label'], test_size=0.20, random_state=42)

#print(messages_bow.shape)

# Create and train the Naive Bayes CLassifier

classifier = MultinomialNB().fit(X_train, y_train)


# Evaluate the model on the training data set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion', methods=['GET'])
def emotion():
    pred = classifier.predict(['ham','Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030'])
    print(pred)

    return('hello')


 
app.run(debug=True)
