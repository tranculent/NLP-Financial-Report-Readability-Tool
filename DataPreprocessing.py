import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize

import os

from itertools import islice

"""***Data Preprocessing tools***"""

class CSVManager:
    def __init__(self, csv_name, directory):
        print("Loading csv file...")
        self.dataset = pd.read_csv(csv_name)
        self.directory = directory
        # self.header = np.concatenate((self.dataset.head(0).columns[:11], self.dataset.head(0).columns[13:]))
        self.ari_scores = []
        self.corpus, self.y = self.extract_columns()
        self.y = self.reshape_columns(self.y)
        
    def extract_columns(self):
        print("     Extracting columns...")
        self.dataset = self.dataset.dropna(how='all')
        y = []
        corpus = []
        c = 0
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
        actual_scores = []
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
# ------ARI-------
                            #print("Folder: " + folder)
                            #print("Text file: " + text_file)
                            f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)
                            # print(sent_tokenize(f.read()))
                            # print(f.read().splitlines())
                            sentences = f.read().splitlines()
                            f.close()
                            # sentences = corpus_reader.sents()
                            # sentences_num = len(sentences)
                            # retrieve the number of characters
                            chars = len([char for sentence in sentences for word in sentence for char in word])
                            #print("Chars: " + str(chars))
                            # retrieve the number of words
                            words = len([word for sentence in sentences for word in sentence])
                            #print("Words: " + str(words))
                            # retrieve the number of sentences
                            
                            #print("Sentences: " + str(sentences))
                            try:
                                formula = 4.71 * (chars / words) + 0.5 * (words / len(sentences)) - 21.43
                                if formula < 5:
                                    self.ari_scores.append(5)
                                elif formula > 14:
                                    self.ari_scores.append(14)
                                else:
                                    self.ari_scores.append(math.floor(formula)) # Automated readability index formula
                            except ZeroDivisionError:
                                self.ari_scores.append(5)
                            
                            if not np.isnan(self.dataset.iloc[i,11]):
                                if math.floor(self.dataset.iloc[i, 11]) in self.flesch_score_table:
                                    actual_scores.append(self.flesch_score_table[math.floor(self.dataset.iloc[i, 11])])
                                else:
                                    actual_scores.append(self.flesch_score_table[99])
                            else:
                                actual_scores.append("N/A")
# ------ARI-------
                            f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)
                            contents = re.sub('[^a-zA-Z]', ' ', f.read()).lower().split()
                            #ps = PorterStemmer()
                            contents = [word for word in contents]
                            corpus.append(' '.join(contents))
                            y.append(self.dataset.iloc[i,12])
                            f = f.close()
                            c += 1
        
        print("Producing excel sheet...")

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
        
        import xlwt
        from xlwt import Workbook

        wb = Workbook()
        sheet1 = wb.add_sheet("Scores")
        final_ari = [ARI_table[ari] for ari in self.ari_scores if ari in ARI_table]
        for i in range(len(final_ari)):
            sheet1.write(i+1, 0, final_ari[i])

        for i in range(len(actual_scores)):
            sheet1.write(i+1, 1, actual_scores[i])

        baseline_score = 0

        for i in zip(final_ari, actual_scores):
            #print(i[0] + " == " + i[1])
            if (i[1] == "College graduate" and i[0] == "Professional") or (i[1] == "College graduate" and i[0] == "College"):
                baseline_score += 1
            if i[0] == i[1]:
                baseline_score += 1

        print("Baseline score: " + str(baseline_score))
        
        wb.save('baseline_scores.xls')
        
        print("Lengths: " + str(len(final_ari)) + " / " + str(len(actual_scores)) + " / " + str(len(self.ari_scores)))

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
        self.ari_scores = np.reshape(self.ari_scores, (len(self.ari_scores), 1))
        print(len(self.ari_scores))
        print(len(y))
        return y

    def get_y_length(self):
        return len(self.y)
    
    def get_y(self):
        return self.y

    def get_corpus(self):
        return self.corpus
    
    def get_ari_scores(self):
        return self.ari_scores

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


class AutomatedReadabilityIndex:
    def __init__(self, csv_name, directory):
        self.dataset = pd.read_csv(csv_name)
        self.directory = directory
    
    def scan_reports(self):
        self.dataset = self.dataset.dropna(how='all')

        ARI_scores = [] # will contain the ARI score for each section of all reports

        # Compare each entry from the flesch table with the ari score

        # Loop through each section of every report
        #