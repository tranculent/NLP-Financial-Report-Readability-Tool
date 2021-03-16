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
from sklearn.metrics import r2_score, mean_squared_error

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import math

from Utils import flesch_score_table

'''***Feature scaling, model training and training evaluation tools.***'''
class Training:
    def __init__(self, corpus, y):
        print("Vectorizing textual data... (ModelTrain.py)")

        self.cv = CountVectorizer(max_features = 30000, stop_words=stopwords.words('english'))
        self.X = self.cv.fit_transform(corpus).toarray()

        print("Labels (ModelTrain.py): " + str(y))

        self.y = self.remove_nans(y) 
        self.y = y

        # printing relevant information
        print("Corpus (vectorized) (ModelTrain.py): " + str(self.X))
        print("Length of Corpus entries (ModelTrain.py): " + str(len(self.X)))
        print("Length of labels (ModelTrain.py): " + str(len(self.y)))
        print("Splitting data into train/test sets... (ModelTrain.py)")

        # Splitting the data into all the required data sets
        # The sets are: X_train (the training data), X_test (the testing data) and y_train (the label training data), y_test (the label testing data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)
        
        self.model = self.init_model()
        self.y_pred = self.evaluate_model()
    
    def remove_nans(self, y):
        for row in y:
            if np.isnan(row[0]):
                row[0] = 0

            row[0] = math.ceil(row[0]*100)/100 # round to 2 decimals after floating point
        return y

    def transform_corpus(self, corpus):
        result = []
        for report in corpus:
            for section in corpus[report]:
                for text_file in section:
                    result.append(section[text_file])
        
        return result

    def init_model(self):
        print("Initializing model... (ModelTrain.py)")

        # Initializes a Decision Tree Regressor (uncomment to test)
        # model = DecisionTreeRegressor(random_state = 0)
        # model.fit(self.X_train, self.y_train)

        # Initializes a Linear Regressor (uncomment to test)
        # model = linear_model.LinearRegression()
        # model.fit(self.X_train, self.y_train)

        # from sklearn.preprocessing import PolynomialFeatures
        # from sklearn.linear_model import LinearRegression
        # poly = PolynomialFeatures(degree=4)
        # X_poly = poly.fit_transform(self.X_train)
        # model = LinearRegression()
        # model.fit(X_poly, self.y_train)

        model = RandomForestRegressor(n_estimators = 10, random_state = 0)
        model.fit(self.X_train, self.y_train)

        return model

    def evaluate_model(self):
        print("Evaluating model... (ModelTrain.py)")

        np.set_printoptions(precision=2)
        y_pred = self.model.predict(self.X_test)
        
        # prints the predicted and test values with only 2 numbers after the floating point (e.g. [0.00, 0.00])
        print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        
        from pandas import DataFrame

        scores = []
        for i in range(len(y_pred)):
            if math.floor(y_pred[i]) in flesch_score_table and math.floor(self.y_test[i]) in flesch_score_table:
                if flesch_score_table[math.floor(y_pred[i])] == flesch_score_table[math.floor(self.y_test[i])]:
                    # predicted correctly
                    scores.append(100) 
                else:
                    # not predicted correctly
                    scores.append(0) 

        # scales the result to be from 1 to 100
        final_score = sum(scores)/len(scores) 

        print("Accuracy score of this model is (ModelTrain.py): " + str(final_score))

        return y_pred

    def feature_scale(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.fit_transform(self.X_test)

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
