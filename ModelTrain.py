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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

import matplotlib.pyplot as plt
from matplotlib import interactive

import math

'''***Feature scaling, model training and training evaluation tools.***'''
class Training:
    def __init__(self, corpus, y):
        print("Vectorizing textual data...")
        self.cv = CountVectorizer(max_features = 30000, stop_words=stopwords.words('english'))
        self.X = self.cv.fit_transform(corpus).toarray()
        print("y: " + str(y))
        self.y = self.remove_nans(y) 
        self.y = y
        print("self.X : " + str(self.X))
        print(len(self.X))
        print(len(self.y))
        print("Splitting into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)
        print("Plotting data...")
        # interactive(True)
        # plt.plot(self.X, self.y)
        # plt.show()
        print("X_train: " + str(self.X_train))
        print("y_train: " + str(self.y_train))
        # self.feature_scale()
        self.model = self.init_model()
        self.y_pred = self.evaluate_model()
    
    def remove_nans(self, y):
        print("Y: " + str(y))
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
        print("Initializing model...")
        model = DecisionTreeRegressor(random_state = 0)
        model.fit(self.X_train, self.y_train)
        # model = linear_model.LinearRegression()
        # model.fit(self.X_train, self.y_train)

        return model

    def evaluate_model(self):
        print("Evaluating model...")
        np.set_printoptions(precision=2)
        y_pred = self.model.predict(self.X_test)
        print('Mean squared error (MSE): %.2f' + str(mean_squared_error(self.y_test, y_pred)))
        print("R2 score: " + str(r2_score(self.y_test, y_pred)))
        print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        #cm = confusion_matrix(self.y_test, y_pred)
        #print(cm)
        
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