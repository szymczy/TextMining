import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

### cz.1
tweet = "    1I enjoyd the event which took place yesteday & I lovdddd itttt ! <http>The link to the show is http://t.co/4ftYom0i It's awesome you'll luv it #HadFun #Enjoyed BFN GN :)"

def cleaning_text(tweet):
    tweet = tweet.lower().strip()
    #zamiana liter na małe oraz usuniecie nadmiernych spacji
    tweet = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|\d", "", tweet)
    #usuniecie interpunkcji oraz emotikon
    stop = stopwords.words('english')
    text = " ".join([word for word in tweet.split() if word not in (stop)])
    #usuniecie stop words
    return text

cleaning_text(tweet)

### cz.2
def stemming_function(sentence):
    porter = PorterStemmer()
    words = word_tokenize(sentence)
    return list(map(porter.stem, words))

s = "Programmers programm in programming languages"
print(stemming_function(s))

