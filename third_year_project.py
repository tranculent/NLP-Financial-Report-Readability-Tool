from DataPreprocessing import CorpusManager
from ModelTrain import Training
import Utils

import os

import numpy as np

# change directory
os.chdir(r"{}".format(Utils.DIRECTORY_PATH))

# get the current directory
directory = os.getcwd()

# initializing the corpus manager
corpus_manager = CorpusManager(Utils.CSV_NAME, directory)

# The file extraction file is not needed for most cases but can be used if the csv file data is not desired.
# dp = FileExtraction(directory) 

training = Training(corpus_manager.get_corpus(), corpus_manager.get_y())

'''
resources:
    https://en.wikipedia.org/wiki/Automated_readability_index
'''
