# TextMining - sentiment analysis of tweets

## About the project:

The purpose of this project is to demonstrate the use of NLP techniques and ML models for sentiment analysis on text data.

This project preprocesses a dataset of tweets about airlines and trains several ML models to classify the sentiment expressed in the tweets as <b>positive, negative or neutral</b>.
Steps performed: 
1. Import of necessary libraries
2. Define functions for preprocessing text:
    - removing HTML tags, punctuation and stop words
    - lemmatizing and stemming words
3. Loads a dataset of tweets from CSV file and applies preprocessing to the text column
4. Display a bar plot showing the number of tweets in each sentiment category
5. Split the preprocessed data into training and testing sets
6. Define 2 ML pipelines: decision tree classifier and random forest classifier
7. Fit each pipeline on the training set and evaluates the models on the testing set using classification report and confusion matrix
8. Display the performance metrics of each model.

## How to install dependencies:
`pip install -r requirements.txt`

## How to execute the project:
`python tweets_analysis.py`
