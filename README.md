# Financial Report Readability Analyzer

## Overview
This project aims to evaluate the readability of annual reports from various companies using advanced Natural Language Processing (NLP) techniques and Machine Learning models. The primary goal is to develop a tool that can automatically analyze and score the readability of these reports, helping stakeholders understand how easily their content can be comprehended by readers. The project involves data preprocessing, model training, and evaluation, utilizing Python as the main programming language.

## Key Features
- **Readability Assessment**: Uses NLP techniques to compute readability scores of financial texts, such as Flesch-Kincaid, Gunning Fog, and more.
- **Machine Learning Models**: Implements various regression and classification models like Decision Trees, Linear Regression, and Random Forest to predict readability scores based on textual features.
- **Data Visualization**: Includes scripts to visualize the distribution of readability scores and the impact of different text features on model predictions.
- **Feature Extraction**: Extracts linguistic features from text such as word length, sentence length, and syllable count to enhance the model's predictive capabilities.

## Contents
- **`DataPreprocessing.py`**: Handles data cleaning, tokenization, feature extraction, and transformation into formats suitable for model training.
- **`ModelTrain.py`**: Script for training machine learning models on processed data and evaluating their performance.
- **`Utils.py`**: Contains utility functions that assist with file handling, data transformation, and metrics calculation.
- **Model Summary Files**: (`decision_tree.txt`, `linear_regression.txt`, `random_forest.txt`) – Contains detailed results and analysis of each model's performance.
- **`main.py`**: The main entry point that integrates all components, running the full readability assessment pipeline.

## Requirements

The zip file in this directory needs to be scanned by the CFIE-FRSE tool.

The CFIE-FRSE tool needs to be placed within the same directory as the src folder.
