import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ML Models imports
from sklearn.linear_model import LinearRegression

# Evaluation imports
from sklearn.metrics import r2_score

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB


'''***Feature scaling, model training and training evaluation tools.***'''
class Training:
    def __init__(self, corpus, y):
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0).
        self.cv = CountVectorizer(max_features = 1500)
        self.X = self.cv.fit_transform(self.transform_corpus(corpus)).toarray()

        print(np.where(np.isnan(y)))
        print(np.where(np.isposinf(y)))
        print(np.where(np.isneginf(y)))
        print(np.where(np.isinf(y)))
        print(y)
        
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)

        self.feature_scale()
        self.model = self.init_model()
        self.y_pred = self.evaluate_model()

    def transform_corpus(self, corpus):
        result = []
        for report in corpus:
            for section in corpus[report]:
                for text_file in section:
                    result.append(section[text_file])

        return result

    def init_model(self):
        # regressor = LinearRegression()
        # regressor.fit(self.X_train, self.y_train)
        # return regressor
        classifier = GaussianNB()
        classifier.fit(self.X, self.y)
        return classifier

    def evaluate_model(self):
        # y_pred = self.model.predict(self.X_test)
        # np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        # print("R2 score: " + str(r2_score(self.y_test, y_pred)))
        y_pred = self.classifier.predict(self.X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        return y_pred

    def feature_scale(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)


    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train
        
    def get_X_test(self):
        return self.X_test
        
    def get_y_test(self):
        return self.y_test

    def get_y_pred(self):
        return self.y_pred