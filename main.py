# Import all the classes
from DataPreprocessing import CorpusManager
from ModelTrain import Training
import Utils

from pathlib import Path
import os

# change directory
# This directory path is custom. To make it suitable for your case, change the DIRECTORY_PATH variable in Utils.py
os.chdir(r"{}".format(Utils.DIRECTORY_PATH))

# get the current directory
directory = os.getcwd()

# intializing the output file which will store the accuracy score of the model
output_file = Path(__file__).with_name(Utils.OUTPUT_FILE_NAME)

# initializing the corpus manager
corpus_manager = CorpusManager(Utils.CSV_NAME, directory, output_file)

# The file extraction file is not needed for most cases but can be used if the csv file data is not desired.
# dp = FileExtraction(directory) 

training = Training(corpus_manager.get_corpus(), corpus_manager.get_y(), output_file)
