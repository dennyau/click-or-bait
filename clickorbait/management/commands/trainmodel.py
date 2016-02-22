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

# redis store dependencies
import redis
import pickle

# django command dependencies
from django.core.management.base import BaseCommand, CommandError

from sklearn.neighbors import KNeighborsClassifier


class Command(BaseCommand):
    help = 'Creates and trains the models used in the Click Or Bait app'

    def handle(self, *args, **options):
        # initialize handler for redis
        redis_server = redis.StrictRedis(host="localhost", port=6379, db=0)

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
        # we found these parameters from our ipython notebook
        vect = CountVectorizer(stop_words='english',ngram_range=(1, 3))

        # fit Naive Bayes and Vectorizer to predict truth rating
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        nb = MultinomialNB()
        nb.fit(X_train_dtm, y_train)

        # Store the CountVectorizer for later use
        with open('count_vectorizer.pickle', 'wb') as handle:
            pickle.dump(vect, handle)
        redis_server.set("model_vect", pickle.dumps(vect))

        # Store the Naive Bayes Model
        with open('naive_bayes.pickle', 'wb') as handle:
            pickle.dump(nb, handle)
        redis_server.set("model_nb", pickle.dumps(nb))


        self.stdout.write(self.style.SUCCESS('Successfully and stored models'))
