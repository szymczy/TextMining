import re
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def main():
    true_news = pd.read_csv(r'News-dataset/True.csv')
    fake_news = pd.read_csv(r'News-dataset/Fake.csv')

    entire_text = ' '.join(true_news['title'].to_list())
    frequencies = count_frequencies(entire_text)
    draw_wordcloud(entire_text, "true_news")

def cleaning_text(tweet):
    tweet = tweet.lower().strip()
    tweet = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|\d", "", tweet)
    stop = stopwords.words('english')
    tweet = " ".join([word for word in tweet.split() if word not in (stop)])
    return tweet


def stemming_function(sentence):
    porter = PorterStemmer()
    words = word_tokenize(sentence)
    return list(map(porter.stem, words))

def count_frequencies(words):
    wordfreq = {}
    for word in words:
        tokens = nltk.tokenize.word_tokenize(word)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    return wordfreq


def draw_wordcloud(bow, plot_title):
    wc = WordCloud()
    wc.generate_from_frequencies(bow)

    plt.axis("off")
    plt.imshow(wc, interpolation='bilinear')
    plt.title(plot_title)
    plt.show()


if __name__ == '__main__':
    main()

