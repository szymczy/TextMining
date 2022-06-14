import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()


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
    a = [i for i in a if len(i) > 3]
    return " ".join(a)


def preprocess(data):
    data['content'] = data['content'].apply(lambda x: x.strip())
    data['clean_title'] = data['title'].apply(
        lambda x: lemmatizer(remove_stop_words(extract_words(x))))
    data['clean_text'] = data['content'].apply(
        lambda x: lemmatizer(remove_stop_words(extract_words(x))))
    data.drop('filename', axis=1, inplace=True)
    return data


def main():
    raw_data = pd.read_csv('bbc-news-data.csv', sep='\t')
    print(raw_data.head())
    print(raw_data.size)

    preprocessed = preprocess(raw_data.copy())
    print(preprocessed.head())
    preprocessed.to_csv('preprocessed.csv', sep='\t')


if __name__ == '__main__':
    main()
