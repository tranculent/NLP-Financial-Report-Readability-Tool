import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re

import nltk
# nltk.download('punkt') UNCOMMENT IF FIRST TIME RUNNING THE PROGRAM (dependency)
# nltk.download('stopwords') UNCOMMENT IF FIRST TIME RUNNING THE PROGRAM (dependency)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize

import os

from itertools import islice

from Utils import flesch_score_table, ARI_table

"""***Data Preprocessing tools***"""

class CorpusManager:
    def __init__(self, csv_name, directory):
        print("Loading csv file...")
        self.dataset = pd.read_csv(csv_name)
        self.directory = directory
        # self.header = np.concatenate((self.dataset.head(0).columns[:11], self.dataset.head(0).columns[13:]))
        self.ari_scores = []
        self.corpus, self.y = self.extract_columns()
        self.y = self.reshape_columns(self.y)
        
    def extract_columns(self):
        print("     Extracting columns... (DataPreprocessing.py)")
        self.dataset = self.dataset.dropna(how='all')
        y = []
        corpus = []
        c = 0

        actual_scores = []

        # Loop through every single row and column from the .csv file.
        for i in range(2, len(self.dataset.iloc[:,:])):
            count = 0

            # Check for eligible rows
            for j in range(len(self.dataset.iloc[i,:])):
                if self.dataset.iloc[i,14] == 1 or self.dataset.iloc[i,16] == 1:
                    if j != 11 and j != 12 and j != 1 and j != 2 and j != 3 and j != 4 and j != 5 and j != 13 and j != 9 and j != 36 and j != 37:
                        if type(self.dataset.iloc[i,j]) is not str and not np.isnan(self.dataset.iloc[i,j]): #front1 = 14, rear1 = 15, front2 = 16, rear2 = 17
                            count += 1
            
            # If more than 14 columns have been correctly defined, proceed to scanning the csv row
            if count > 14:
                folder = self.dataset.iloc[i,1][:-4]
                if folder[-1] == " ":
                    folder = folder[:-1]

                # If the folder for that csv row exists, proceed
                if len(os.listdir(self.directory + "/" + folder)) != 0:
                    # Loop through each section text for the report corresponding to the csv row
                    for text_file in os.listdir(self.directory + "/" + folder + "/" + "sections_text"):
                        t = ""
                        if text_file[1] == "_":
                            t = text_file[2:]
                        elif text_file[2] == "_":
                            t = text_file[3:]
                                                        
                        if t[:-4] == self.dataset.iloc[i,2]:
# ------ARI-------
# Uncomment the following lines for more information about the fields.
                            #print("Folder: " + folder)
                            #print("Text file: " + text_file)
                            f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)
                            sentences = f.read().splitlines()
                            f.close()
                            # retrieve the number of characters
                            chars = len([char for sentence in sentences for word in sentence for char in word])
                            #print("Chars: " + str(chars))
                            # retrieve the number of words
                            words = len([word for sentence in sentences for word in sentence])
                            #print("Words: " + str(words))
                            # retrieve the number of sentences
                            
                            #print("Sentences: " + str(sentences))

                            # Fill the ari_scores ari with the appropriate scores
                            try:
                                # Compute the formula for every section of each report
                                formula = 4.71 * (chars / words) + 0.5 * (words / len(sentences)) - 21.43

                                # IF the formula scored below the minimum threshold, simply add the minimum threshold (avoids scaling issues)
                                if formula < 5:
                                    self.ari_scores.append(5)
                                # IF the formula exceeds 14, add 14 (avoids scaling issues)
                                elif formula > 14:
                                    self.ari_scores.append(14)
                                else:
                                    # Formula is fine, simply add it as it is
                                    self.ari_scores.append(math.floor(formula)) # Automated readability index formula
                            except ZeroDivisionError:
                                self.ari_scores.append(5)
                            
                            # Check if the Flesch score column is not NaN
                            if not np.isnan(self.dataset.iloc[i,11]):
                                # Checks if the score is not invalid (e.g. out of the range 1-100)
                                if math.floor(self.dataset.iloc[i, 11]) in flesch_score_table:
                                    # Add the score to actual_scores
                                    actual_scores.append(flesch_score_table[math.floor(self.dataset.iloc[i, 11])])
                                else:
                                    # Otherwise add a 99 (avoids scaling issues)
                                    actual_scores.append(flesch_score_table[99])
                            else:
                                actual_scores.append("N/A")
# ------ARI-------
                            # Open the text file representing each section for the report
                            f = open(self.directory + "/" + folder + "/" + "sections_text" + "/" + text_file)

                            # Remove everything that is not a letter with a space and convert to lower case everything.
                            contents = re.sub('[^a-zA-Z]', ' ', f.read()).lower().split()
                            # (Optional) Uncomment to use stemming -> make sure to add ps.stem(word) for the list comprehension below
                            #ps = PorterStemmer()

                            # Initialize the contents using list comprehension
                            contents = [word for word in contents]
                            
                            # Add the contents to the corpus.
                            corpus.append(' '.join(contents))

                            # Add the label.
                            y.append(self.dataset.iloc[i,12])

                            # Close file.
                            f = f.close()

                            # Increase number of sections scanned.
                            c += 1

        # Get the grade levels based on the ARI_table coming from Utils.py.
        final_ari = [ARI_table[ari] for ari in self.ari_scores if ari in ARI_table]

        # Initialize the baseline score.
        baseline_score = 0

        # Loop through both final_ari and actual_scores
        for i in zip(final_ari, actual_scores):
            # Compare each grade level
            # If it matches, add 1 to baseline score
            if (i[1] == "College graduate" and i[0] == "Professional") or (i[1] == "College graduate" and i[0] == "College"):
                baseline_score += 1
            if i[0] == i[1]:
                baseline_score += 1

        print("Baseline score (DataPreprocessing.py): " + str(baseline_score / len(actual_scores) * 100))
        
        print("Lengths (DataPreprocessing.py): " + str(len(final_ari)) + " / " + str(len(actual_scores)) + " / " + str(len(self.ari_scores)))

        print("Count of scanned reports (DataPreprocessing.py): " + str(c))

        return (corpus, y)
    
    # Checks if the row is valid.
    def is_valid(self, i):
        for j in range(len(self.dataset.iloc[i, :])):
            if type(self.dataset.iloc[i,j]) != str and j != 9 and j != 13:
                if np.isnan(self.dataset.iloc[i,j]):
                    return False
        return True

    # Reshapes the 'y' to be in specific shape that is required by ML models.
    def reshape_columns(self, y):
        y = np.reshape(y, (len(y), 1))
        self.ari_scores = np.reshape(self.ari_scores, (len(self.ari_scores), 1))
        print(len(self.ari_scores))
        print(len(y))
        return y

    """ Getter Methods """
    def get_y_length(self):
        return len(self.y)
    
    def get_y(self):
        return self.y

    def get_corpus(self):
        return self.corpus
    
    def get_ari_scores(self):
        return self.ari_scores

class FileExtraction:
    '''
    This class has been deprecated as the CorpusManager uses the information here plus utilizing the csv file. However, if for some reason the .csv file will not be used and only the
    textual contents of the reports will be required, this file can be used.
    '''
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
        # Uncomment for more info (note that it will take a lot of printing space)
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
