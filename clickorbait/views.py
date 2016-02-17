# Data Science Requirements
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

# Web App Requirements
from django.template import loader
from django.http import HttpResponse


def index(request):
    # read clickbait.csv into a DataFrame
    # documentation: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    url = 'clickorbait/data/clickbait.csv'
    column_names = ["url","title","classification"]
    clickbait = pd.read_csv(url,names=column_names,header=None)

    # convert label to a numeric variable
    clickbait['truth'] = clickbait.classification.map({'Fake':0, 'Truth':1})

    context = {}
    context['sample_table'] = clickbait.head().to_html()

    # define X and y
    X = clickbait.title
    y = clickbait.truth

    # split the new DataFrame into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # use CountVectorizer to create document-term matrices from X_train and X_test
    # include 1-grams and 2-grams, we end up with many features
    vect = CountVectorizer(ngram_range=(1, 2))
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # rows are documents, columns are terms (aka "tokens" or "features")
    context['training_shape'] = "%s" % (X_train_dtm.shape,)
    context['features_sample'] = '[%s]' % ', '.join(map(str, vect.get_feature_names()[-50:]))

    # create document-term matrices
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # use Naive Bayes to predict the truth rating
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)

    # calculate accuracy
#    context['accuracy'] = str(metrics.accuracy_score(y_test, y_pred_class))

    test_test_dtm = vect.transform(['Top 10 hot chicks to see right now'])
    context['accuracy'] = str(nb.predict(test_test_dtm))

    # calculate null accuracy
    y_test_binary = np.where(y_test==5, 1, 0)
    context['null_accuracy'] = str(max(y_test_binary.mean(), 1 - y_test_binary.mean()))

    # define a function that accepts a vectorizer and calculates the accuracy
    def tokenize_test(vect):
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        nb = MultinomialNB()
        nb.fit(X_train_dtm, y_train)
        y_pred_class = nb.predict(X_test_dtm)
        return 'Features: '+ str(X_train_dtm.shape[1]) + '<br/>Accuracy: ' +  str(metrics.accuracy_score(y_test, y_pred_class))

    # remove English stop words, ngram: 1-3
    context['extras'] = tokenize_test(CountVectorizer(stop_words='english',ngram_range=(1, 3)))

    template = loader.get_template('clickorbait/basis.html')

    return HttpResponse(template.render(context, request))
