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

from itertools import islice

"""***Data Preprocessing tools***"""

class CSVManager:
    def __init__(self, csv_name, directory):
        print("Loading csv file...")
        self.dataset = pd.read_csv(csv_name)
        self.directory = directory
        # self.header = np.concatenate((self.dataset.head(0).columns[:11], self.dataset.head(0).columns[13:]))
        self.corpus, self.y = self.extract_columns()
        self.y = self.reshape_columns(self.y)
        
    def extract_columns(self):
        print("     Extracting columns...")
        self.dataset = self.dataset.dropna(how='all')
        y = []
        corpus = []
        c = 0
        for i in range(2, len(self.dataset.iloc[:,:])):
            count = 0
            # Check for eligible rows
            for j in range(len(self.dataset.iloc[i,:])):
                if self.dataset.iloc[i,14] == 1 or self.dataset.iloc[i,16] == 1:
                    if j != 11 and j != 12 and j != 1 and j != 2 and j != 3 and j != 4 and j != 5 and j != 13 and j != 9 and j != 36 and j != 37:
                        if type(self.dataset.iloc[i,j]) is not str and not np.isnan(self.dataset.iloc[i,j]): #front1 = 14, rear1 = 15, front2 = 16, rear2 = 17
                            count += 1
            
            if count > 14:
                folder = self.dataset.iloc[i,1][:-4]
                if folder[-1] == " ":
                    folder = folder[:-1]

                if len(os.listdir(self.directory + "/" + folder)) != 0:
                    for text_file in os.listdir(self.directory + "/" + folder + "/" + "sections_text"):
                        t = ""
                        if text_file[1] == "_":
                            t = text_file[2:]
                        elif text_file[2] == "_":
                            t = text_file[3:]
                                                        
                        if t[:-4] == self.dataset.iloc[i,2]:
                            f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)
                            contents = re.sub('[^a-zA-Z]', ' ', f.read()).lower().split()
                            # ps = PorterStemmer()
                            # contents = [ps.stem(word) for word in contents if word not in stopwords.words('english')]
                            contents = [word for word in contents if word not in stopwords.words('english')]
                            corpus.append(' '.join(contents))
                            y.append(self.dataset.iloc[i,12])
                            f = f.close()
                            c += 1
                        
        print("Count: " + str(c))
        return (corpus, y)
    
    def is_valid(self, i):
        for j in range(len(self.dataset.iloc[i, :])):
            if type(self.dataset.iloc[i,j]) != str and j != 9 and j != 13:
                if np.isnan(self.dataset.iloc[i,j]):
                    return False
        return True

    def reshape_columns(self, y):
        y = np.reshape(y, (len(y), 1))
        return y

    def get_y_length(self):
        return len(self.y)
    
    def get_y(self):
        return self.y

    def get_corpus(self):
        return self.corpus

class FileExtraction:
    def __init__(self, directory):
        self.directory = directory
        self.corpus = {}
    
        self.extract_sections()
        #self.bag_of_words = {}
        #self.extract_words_counts()
        self.print_corpus_info()

        # self.result_corpus_x_y = {}

    def is_integer(self, n):
        try:
            int(n)
            return True
        except ValueError:
            return False

    def get_words_counts(self):
        return self.bag_of_words

    def print_corpus_info(self):
        # print("Corpus keys: " + str(self.corpus.keys()))
        print("Number of keys: " + str(len(self.corpus.keys())))
        # print("First two keys: " + str(dict(islice(self.corpus.items(), 2))))
        # print("Number of items: " + str(len(self.corpus.items())))
    
    def extract_sections(self):
        print("Extracting sections from pdfs...")
        count = 0
        for folder in os.listdir(self.directory):
            if not os.path.isfile(folder):
                if len(os.listdir(self.directory + "/" + str(folder))) != 0:
                    self.corpus[folder] = []
                    for text_file in os.listdir(self.directory + "/" + str(folder) + "/" + "sections_text"):
                        f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)
                        contents = re.sub('[^a-zA-Z]', ' ', f.read()).lower().split()
                        ps = PorterStemmer()
                        contents = [ps.stem(word) for word in contents if word not in stopwords.words('english')]
                        temp_dict = {}
                        temp_dict[text_file] = ' '.join(contents)
                        self.corpus[folder].append(temp_dict)
                        count += 1
                        f = f.close()
        print("Count: " + str(count))

    def extract_words_counts(self):
        print("extracting word counts from each section...")
        for report in self.corpus:
            for section in self.corpus[report]:
                for text_file in section:
                    if not self.is_integer(section[text_file]) and len(section[text_file]) > 1:
                        if section[text_file] in self.bag_of_words:
                            self.bag_of_words[section[text_file]] = self.bag_of_words.get(section[text_file]) + 1
                        else:
                            self.bag_of_words[section[text_file]] = 1

        self.bag_of_words = {key: value for key, value in sorted(self.bag_of_words.items(), key=lambda item: item[1], reverse=True)}

    def get_corpus(self):
        return self.corpus