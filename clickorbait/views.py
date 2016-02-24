# Data Science Requirements
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import StringIO
import base64
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

# redis store dependencies
import redis
import pickle

# Web App Requirements
from django.template import loader
from django.http import HttpResponse
import sys

def predict(request):
    context = {}
    query_title = request.POST.get('query-title', '')
    if (query_title):
        # use Naive Bayes model from Redis  to predict the truth rating
        redis_server = redis.StrictRedis(host="localhost", port=6379, db=0)
        nb = pickle.loads(redis_server.get("model_nb"))
        vect = pickle.loads(redis_server.get("model_vect"))

        test_test_dtm = vect.transform([query_title])
        q_a = nb.predict(test_test_dtm).tolist()
        #print >>sys.stderr, q_a
        context['query_answers'] = q_a

    template = loader.get_template('clickorbait/prediction.html')

    return HttpResponse(template.render(context, request))

def sentiment_boxplot(request):
    context = {}

    # draw the sentiment analysis boxplot - reference https://gist.github.com/tebeka/5426211
    # fetch pandas dataframe from redis
    redis_server = redis.StrictRedis(host="localhost", port=6379, db=0)
    pd = pickle.loads(redis_server.get("clickbait_dataframe"))

    # make a matplotlib object to store the plot in
    figure = plt.figure()
    axis = figure.add_subplot(1,1,1)
    pd.boxplot(column='sentiment', by='truth', ax=axis)

    # convert the plot into a base64 string representing the image
    io = StringIO.StringIO()
    figure.savefig(io,format='png')
    context['png_base64'] = base64.encodestring(io.getvalue())

    template = loader.get_template('clickorbait/imageBase64.html')

    return HttpResponse(template.render(context, request))

def index(request):
    context = {}

    # fetch pandas dataframe from redis
    redis_server = redis.StrictRedis(host="localhost", port=6379, db=0)
    clickbait = pickle.loads(redis_server.get("clickbait_dataframe"))
    
    # fetch Vectorizer from redis
    vect = pickle.loads(redis_server.get("model_vect"))
    features = vect.get_feature_names()
    dtm = vect.fit_transform(clickbait.title)

    # choose a random title that is at least 50 characters
    linkTitle_length = 0
    while linkTitle_length < 50:
        linkTitle_id = np.random.randint(0, len(clickbait))
        linkTitle_text = unicode(clickbait.title[linkTitle_id], 'utf-8')
        linkTitle_length = len(linkTitle_text)
    
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in TextBlob(linkTitle_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[linkTitle_id, features.index(word)]
    
    # save words with the top 5 TF-IDF scores
    context['top_words_title'] = linkTitle_text;
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    context['top_scores'] = top_scores

    template = loader.get_template('clickorbait/basis.html')
    return HttpResponse(template.render(context, request))

def index2(request):
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
