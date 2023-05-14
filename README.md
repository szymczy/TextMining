# Classification of news using bbc dataset

## About the project:
Project consists of 2 files: preprocessing and data_exploration.

In the first file, text preprocessing on a dataset of news articles stored in csv 'bbc-news-data' is performed.

The script uses various NLP techniques to preprocess the text:
* converting the text to lowercase
* removing HTML tags
* removing punctuation marks
* removing stopwords (commonly used words like 'a', 'an', 'the' etc.)
* lemmatizing the text (reducing words to their base form)


Second file is a script for text classification using various ML models. 

It visualizes the distribution of articles across categories using a bar plot and creates word clouds for each category. 
The dataset is split into training and test sets.
Various ML models are fitted to the training set, using different feature extraction techniques as TF-IDF and CountVectorizer, and different classification algorithms such as Random Forest, Decision Tree, SVM and Multinominal Naive Bayes.

For each model, the script print a classification report and a confusion matrix. 

In the end, the trained models are saved as pickle files.


## How to install dependencies:

`pip install -r requirements.txt`

## How to execute preprocessing:

`python preprocessing.py`

## How to execute the project:
File `data_exploration.ipynb` should be opened in Jupyter Notebook environment or execute the script:
`python data_exploration.py`
