import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
#from wordcloud import WordCloud #import?
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

wl = WordNetLemmatizer()
stemmer = PorterStemmer()

def extract_words(text):
    text = text.lower().strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_stop_words(text):
    words = [i for i in text.split() if i not in stopwords.words('english')]
    return ' '.join(words)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatizer(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(text))
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1]))
         for idx, tag in enumerate(word_pos_tags)]
    return a

def stem(text):
    a = [stemmer.stem(x) for x in text]
    return ' '.join(a)

def preprocess(data):
    data['clean_text'] = data['text'].apply(lambda x: x.strip())
    data['clean_text'] = data['clean_text'].apply(
        lambda x: stem(lemmatizer(remove_stop_words(extract_words(x)))))
    return data

def main():
    raw_data = pd.read_csv('tweets_airline.csv', sep=',', usecols=['text', 'airline_sentiment'])
    print(raw_data.head())
    print(raw_data.columns)
    print(raw_data.size)

    preprocessed = preprocess(raw_data.copy())
    print(preprocessed.head())
    preprocessed.to_csv('preprocessed.csv', sep=',')
    data = pd.read_csv('preprocessed.csv', sep=',', usecols=['text', 'airline_sentiment', 'clean_text'])
    print(data.head())
    categories = data['airline_sentiment'].value_counts()
    bar_plot = plt.bar(categories.keys(), categories.values, align='center', color=['red', 'green', 'blue'])
    plt.title('Number of tweets by category')
    plt.xlabel('Category')
    plt.ylabel('Number')
    plt.show()

    X = data['clean_text']
    Y = data['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

    pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', DecisionTreeClassifier())])

    model = pipeline.fit(X_train, y_train)
    print('DECISION TREE')
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))

    pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', RandomForestClassifier())])

    model = pipeline.fit(X_train, y_train)
    print('RANDOM FOREST')
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))
#RANDOM FOREST ma wyzszy poziom dokladnosci w przewidywaniu sentymentu

    
if __name__ == '__main__':
    main()
