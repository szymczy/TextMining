import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import preprocessing
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('preprocessed.csv', sep='\t', index_col=0)
print(data.head())

categories = data['category'].value_counts()
bar_plot = plt.bar(categories.keys(), categories.values, align='center', color=[
                   'red', 'green', 'blue', 'cyan', 'magenta'])
plt.title('Number of articles by category')
plt.xlabel('Category')
plt.ylabel('Number')
plt.yticks(np.arange(0, 600, step=100))
plt.show()

sport = data[data['category'] == 'sport']['clean_text']
business = data[data['category'] == 'business']['clean_text']
politics = data[data['category'] == 'politics']['clean_text']
tech = data[data['category'] == 'tech']['clean_text']
entertainment = data[data['category'] == 'entertainment']['clean_text']


def plot_wordcloud(words, title):
    plt.figure()
    wordcloud = WordCloud(max_font_size=200, background_color='black').generate(words.str.cat(sep=' '))
    plt.imshow(wordcloud)
    plt.yticks([])
    plt.xticks([])
    plt.title(title)
    plt.show()


plot_wordcloud(sport, "Sport")
plot_wordcloud(business, "Business")
plot_wordcloud(tech, "Tech")
plot_wordcloud(politics, "Politics")
plot_wordcloud(entertainment, "Entertainment")

X = data['clean_text']
Y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


#RandomForest
def print_report(model, y_test, model_name):
    print('\nModel name: ' + model_name)
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))
    print('')


pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', RandomForestClassifier())])

model = pipeline.fit(X_train, y_train)
with open('RandomForest_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'RandomForestClassifier with TfidfVectorizer')

pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', RandomForestClassifier())])

model = pipeline.fit(X_train, y_train)
with open('RandomForest_CV.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'RandomForestClassifier with CountVectorizer')

#DecisionTree
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', DecisionTreeClassifier())])

model = pipeline.fit(X_train, y_train)
with open('DecisionTree_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'DecisionTreeClassifier with TfidfVectorizer')

pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', DecisionTreeClassifier())])

model = pipeline.fit(X_train, y_train)
with open('DecisionTree_CV.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'DecisionTreeClassifier with CountVectorizer')

# SVC
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=3000)),
                     ('clf', SVC())])

model = pipeline.fit(X_train, y_train)
with open('SVC_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'SVC with TfidfVectorizer')

pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=3000)),
                     ('clf', SVC())])

model = pipeline.fit(X_train, y_train)
with open('SVC_CV.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'SVC with CountVectorizer')

# MultinomialNB
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', MultinomialNB())])

model = pipeline.fit(X_train, y_train)
with open('MultinomialNB_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'MultinomialNB with TfidfVectorizer')

pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', MultinomialNB())])

model = pipeline.fit(X_train, y_train)
with open('MultinomialNB_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'MultinomialNB with CountVectorizer')

# AdaBoostClassifier
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', AdaBoostClassifier())])

model = pipeline.fit(X_train, y_train)
with open('AdaBoostClassifier_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'AdaBoostClassifier with TfidfVectorizer')

pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', AdaBoostClassifier())])

model = pipeline.fit(X_train, y_train)
with open('AdaBoostClassifier_CV.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'AdaBoostClassifier with CountVectorizer')

# BaggingClassifier
pipeline = Pipeline([('vect', TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', BaggingClassifier())])

model = pipeline.fit(X_train, y_train)
with open('BaggingClassifier_TFIDF.pickle', 'wb') as f:
    pickle.dump(model, f)
print_report(model, y_test, 'BaggingClassifier with TfidfVectorizer')

# 100 CountVectorizer tokens
count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
chi = SelectKBest(chi2, k=100)
vectorized = count_vectorizer.fit_transform(X_train)
selected = chi.fit_transform(vectorized, y_train)
mask = chi.get_support()

print(np.array(count_vectorizer.get_feature_names())[mask])

# 100 TfidfVectorizer tokens
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
chi = SelectKBest(chi2, k=100)
vectorized = tfidf_vectorizer.fit_transform(X_train)
selected = chi.fit_transform(vectorized, y_train)
mask = chi.get_support()

print(np.array(tfidf_vectorizer.get_feature_names())[mask])
article1 = '''The PGA Tour has suspended all of its members who are playing in this week's Saudi-funded LIV Golf Invitational.
The tournament, being held at Centurion Club near London, is the most lucrative in the history of the game with a $25m (£20m) prize fund.
Six-time major winner Phil Mickelson is among the biggest names to be affected.
LIV Golf issued an immediate reply calling the PGA Tour "vindictive" and said this "deepens the divide between the Tour and its members".
LIV Golf added: "It's troubling that the Tour, an organisation dedicated to creating opportunities for golfers to play the game, is the entity blocking golfers from playing.
This certainly is not the last word on this topic. The era of free agency is beginning as we are proud to have a full field of players joining us in London, and beyond."
The PGA Tour released its statement 30 minutes after play had begun at Centurion, where 48 players are competing in the first of eight planned LIV Golf invitational events.
The first seven events all have the same $25m prize fund, with the winner collecting $4m, while a team element will see the top three teams share $5m. The eighth event is a $50m team championship and will be played in October at Trump Doral in Miami.
The PGA Tour had refused requests for waivers from its members wanting to play in the new series and had threatened to ban those that rebelled.
This decision affects 17 players, including lifetime member Mickelson and his fellow American Dustin Johnson, who announced on Tuesday that he had resigned from the Tour.
Crucially, the PGA Tour has closed a loophole that would potentially have allowed players such as Johnson, as well as Sergio Garcia, Martin Kaymer, Graeme McDowell and Lee Westwood, who had all also already resigned from the PGA Tour, to play in events via sponsor's exemptions.
The PGA Tour said: "In accordance with the PGA Tour's tournament regulations, the players competing this week without releases are suspended or otherwise no longer eligible to participate in PGA Tour tournament play, including the Presidents Cup.
The same fate holds true for any other players who participate in future Saudi Golf League events in violation of our regulations.
These players have made their choice for their own financial-based reasons," said PGA Tour commissioner Jay Monahan in the statement that was issued to all members of the tour.
But they can't demand the same PGA Tour membership benefits, considerations, opportunities and platform as you. That expectation disrespects you, our fans and our partners.
The players will be removed from the FedEx Cup points list following the end of the Canadian Open on Sunday.
The European-based DP World Tour, when contacted by BBC Sport, declined to comment.'''

article2 = '''Of the 8,300 million tonnes of virgin plastic produced up to the end of 2015, 6,300 million tonnes has been discarded. Most of that plastic waste is still with us, entombed in landfills or polluting the environment. Microplastics have been found in Antarctic sea ice, in the guts of animals that live in the deepest ocean trenches, and in drinking water around the world. In fact, plastic waste is now so widespread that researchers have suggested it could be used as a geological indicator of the Anthropocene.
But what if we could wave a magic wand and remove all plastics from our lives? For the sake of the planet, it would be a tempting prospect – but we'd quickly find out just how far plastic has seeped into every aspect of our existence. Is life as we know it even possible without plastic
Humans have been using plastic-like materials, such as shellac – made from a resin secreted by lac insects – for thousands of years. But plastics as we know them today are a 20th Century invention: Bakelite, the first plastic made from fossil fuels, was invented in 1907. It wasn't until after World War Two that production of synthetic plastics for use outside the military really took off. Since then, plastic production has increased almost every year, from two million tonnes in 1950 to 380 million tonnes in 2015. If it continues at this rate, plastic could account for 20% of oil production by 2050.
Today, the packaging industry is by far the biggest user of virgin plastic. But we also use plastic in plenty of longer-lasting ways too: it's in our buildings, transport, and other vital infrastructure, not to mention our furniture, appliances, TVs, carpets, phones, clothes, and countless other everyday objects.
All this means a world entirely without plastic is unrealistic. But imagining how our lives would change if we suddenly lost access to plastic can help us figure out how to forge a new, more sustainable relationship with it.
In hospitals, the loss of plastic would be devastating. "Imagine trying to run a dialysis unit with no plastic," says Sharon George, senior lecturer in environmental sustainability and green technology at Keele University in the UK.
Imagine trying to run a dialysis unit with no plastic – Sharon George
Plastic is used in gloves, tubing, syringes, blood bags, sample tubes and more. Since the discovery of variant Creutzfeldt–Jakob disease (vCJD) in 1996 – caused by misfolded proteins called prions that can survive normal hospital sterilisation processes – standard reusable surgical instruments have even been replaced by single-use versions for some operations. According to one study, a single tonsillectomy operation in a UK hospital can result in more than 100 separate pieces of plastic waste. While some surgeons have argued that single-use plastic is overused in hospitals, right now many plastic medical items are essential, and lives would be lost without them.
Some everyday plastic items are also vital for protecting health. Condoms and diaphragms are on the World Health Organization's list of essential medicines, and face masks – including plastic-based surgical masks and respirators, as well as reusable cloth masks – have helped slow the spread of the Covid-19 virus. "A mask that you have for Covid is related to our safety and the safety of others," says George. "The impact of taking that away could be loss of life, if you took it away on a big scale.'''

article3 = '''Singapore (CNN)The US seeks "guard rails" with China, according to senior defense officials, in the first meeting between Defense Secretary Lloyd Austin and his Chinese counterpart, as tensions grow over what Washington sees as Beijing's increasingly aggressive actions in the region.
The meeting, scheduled to take place on Friday evening in Singapore, will focus in part on "setting guard rails on the relationship," one official said, while calling for more mature crisis communications mechanisms to ensure that the growing competition between the world's two preeminent world powers does not escalate into conflict.
The upcoming meeting during the IISS's conference is the first between Austin and Minister of National Defense General Wei Fenghe. Despite US focus on the Indo-Pacific as the priority region for the future and calling China the "pacing challenge," Austin has only spoken to Wei once in an April 20 phone call. It was the first such call since the previous administration.
US officials have been negotiating the specific details of the meeting, the official said, with the aim of avoiding the very public spectacle of the first meeting between the US and China under the Biden administration. That meeting, held in March 2021 in Alaska, quickly led to Secretary of State Antony Blinken and his People's Republic of China (PRC) counterpart accusing each other of violating everything from the established rules of the meeting to the international order.
"One of the ground rules that we aim to establish with the PRC is that we're going to characterize our position and they can characterize their position," the official said. "I think we are taking every effort to ensure that this is a professional, substantive meeting."
The meeting comes during Austin's fourth trip to the Indo-Pacific region after a formal request from China's military leadership.
In addition to trying to establish lines of communication at the highest levels of the militaries, the US also wants to see communication mechanisms between commanders at the theater level.
"This has been a priority for us in the defense relationship," the official said. The US also has a "relatively new" crisis communications working group with China, the official said. While there isn't a date set for the next meeting, both sides agree that it should happen this year. Wei emphasized the working group in the call, according to the official.
The US has frequently called out what it views as China's growing aggression in the region, accusing the People's Liberation Army of unsafe and dangerous activity, particularly around the South China Sea and Taiwan.
Australia -- one of America's closest allies in the Indo-Pacific -- condemned Beijing when a Chinese fighter jet released chaffs and flares near an Australian surveillance plane late last month.
At the same time, China has been vocal in its condemnation of relations between the US and Taiwan. After a congressional delegation visited Taiwan late last month, the Chinese embassy in Washington urged the US in to "avoid sending wrong signals to the 'Taiwan independence' separatist forces," according to a statement from the embassy. That same week, China sent 30 warplanes into Taiwan's air defense identification zone, the highest daily figure in four months.
"The Taiwan issue will figure prominently in all of the secretary's conversations," the senior defense official said.'''


processed_article = preprocessing.lemmatizer(preprocessing.remove_stop_words(preprocessing.extract_words(article1)))
print(model.predict(pd.Series(data={'clean_text': processed_article})))

processed_article = preprocessing.lemmatizer(preprocessing.remove_stop_words(preprocessing.extract_words(article2)))
print(model.predict(pd.Series(data={'clean_text': processed_article})))

processed_article = preprocessing.lemmatizer(preprocessing.remove_stop_words(preprocessing.extract_words(article3)))
print(model.predict(pd.Series(data={'clean_text': processed_article})))