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

import math

'''***Feature scaling, model training and training evaluation tools.***'''
class Training:
    def __init__(self, corpus, ari_scores, y):
        print("Vectorizing textual data...")

        self.cv = CountVectorizer(max_features = 30000, stop_words=stopwords.words('english'))
        self.X = self.cv.fit_transform(corpus).toarray()
        self.ari_scores = ari_scores # ARI SCORE 

        print("y: " + str(y))

        self.y = self.remove_nans(y) 
        self.y = y

        print("self.X : " + str(self.X))
        print(len(self.X))
        print(len(self.y))
        print("Splitting into train/test sets...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)

        print("Plotting data...")
        
        self.model = self.init_model()
        self.y_pred = self.evaluate_model()
        # self.baseline()
    
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
        # model = DecisionTreeRegressor(random_state = 0)
        # model.fit(self.X_train, self.y_train)
        model = linear_model.LinearRegression()
        model.fit(self.X_train, self.y_train)

        return model

    def evaluate_model(self):
        print("Evaluating model...")
        np.set_printoptions(precision=2)
        y_pred = self.model.predict(self.X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

        self.flesch_score_table = {}
        for i in range(100):
            if i < 10:
                self.flesch_score_table[i] = "Professional"
            elif i < 30:
                self.flesch_score_table[i] = "College graduate"
            elif i < 50:
                self.flesch_score_table[i] = "College"
            elif i < 60:
                self.flesch_score_table[i] = "10th to 12th grade"
            elif i < 70:
                self.flesch_score_table[i] = "8th & 9th grade"
            elif i < 80:
                self.flesch_score_table[i] = "7th grade"
            elif i < 90:
                self.flesch_score_table[i] = "6th grade"
            elif i < 100:
                self.flesch_score_table[i] = "5th grade"
        
        from pandas import DataFrame
        
        test_vals_1d = [entry[0] for entry in self.y_test]

        scores = []
        for i in range(len(y_pred)):
            if math.floor(y_pred[i]) in self.flesch_score_table and math.floor(self.y_test[i]) in self.flesch_score_table:
                if self.flesch_score_table[math.floor(y_pred[i])] == self.flesch_score_table[math.floor(self.y_test[i])]:
                    scores.append(100) # predicted correctly
                else:
                    scores.append(0) # not predicted correctly

        final_score = sum(scores)/len(scores)
        print("Accuracy score of this model is: " + str(final_score))

        # Formula for normalization: 100 * (max - value) / range
        
        # normalize the accuracy score to be in the range of 1-100
        return y_pred

    def baseline(self):
        # Only use the ARI scores that start from 5
        # If the ARI score is less than 5, then automatically count it out as a wrong
        print("Evaluating baseline score..")
        ARI_table = {
            5: "5th grade",
            6: "6th grade",
            7: "7th grade",
            8: "8th & 9th grade",
            9: "8th & 9th grade",
            10: "10th to 12th grade",
            11: "10th to 12th grade",
            12: "10th to 12th grade",
            13: "College",
            14: "Professional"
        }

        baseline_score = 0
        for i in range(len(self.y_test)):
            score = ""
            if self.ari_scores[i] > 14:
                score = "Professional"
            elif self.ari_scores[i] < 5:
                continue
            else:
                score = ARI_table[self.ari_scores[i][0]]

            if math.floor(self.y_test[i]) in self.flesch_score_table:
                if score == self.flesch_score_table[math.floor(self.y_test[i])]:
                    baseline_score += 1
        
        print ("Baseline Scores")
        print(baseline_score)
        print("Length of ari_scores (can be used for rescaling to 1:100): " + str(len(self.ari_scores)))
        print("Length of y_test: " + str(len(self.y_test)))
        
        # print(baseline_score / 10)
        # print(baseline_score / 100)

        print("Length of y test: " + str(len(self.y_test)))

        print("ari_scores[i]: " + str(self.ari_scores[0]))
        print("ari_scores[i][j]: " + str(self.ari_scores[0][0]))

        return baseline_score

        # Evaluate a score from 1-100 based on the correct scores from the ARI table compared to the flesch_score table

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