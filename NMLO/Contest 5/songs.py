import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    html_free = soup.get_text()
    return html_free
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text
stemmer = PorterStemmer()
def word_stemmer(text):
    stem_text = [stemmer.stem(i) for i in text]
    return stem_text

tokenizer = RegexpTokenizer(r'\w+')
df_train['lyric'] = df_train['lyric'].apply(lambda x: remove_punctuation(remove_html(x)))
df_train['lyric'] = df_train['lyric'].apply(lambda x: tokenizer.tokenize(x.lower()))
df_train['lyric'] = df_train['lyric'].apply(lambda x: remove_stopwords(x))
df_train['lyric'] = df_train['lyric'].apply(lambda x: word_lemmatizer(x))
df_train['lyric'] = df_train['lyric'].apply(lambda x: word_stemmer(x))
df_test['lyric'] = df_test['lyric'].apply(lambda x: remove_punctuation(remove_html(x)))
df_test['lyric'] = df_test['lyric'].apply(lambda x: tokenizer.tokenize(x.lower()))
df_test['lyric'] = df_test['lyric'].apply(lambda x: remove_stopwords(x))
df_test['lyric'] = df_test['lyric'].apply(lambda x: word_lemmatizer(x))
df_test['lyric'] = df_test['lyric'].apply(lambda x: word_stemmer(x))
train = df_train.values
test = df_test['lyric'].values

word_dict = {}
for wordlist, point in train:
    point = (point-.5)*2
    for word in wordlist:
        if word in word_dict: word_dict[word] = (word_dict[word][0]+point, word_dict[word][1]+1)
        else: word_dict[word] = (point, 1)

from sklearn.neural_network import MLPClassifier

training = []
for wordlist, point in train:
    words = []
    for word in wordlist:
        if word in word_dict: words.append(word_dict[word][0] / word_dict[word][1])
        elif '*' in word: words.append(-1)
        elif set(word) & set('1234567890'): words.append(-.5)
    while len(training) < 20: words.append(0)
    training.append(words)

results = []
for wordlist in test:
    words = []
    for word in wordlist:
        print(word)
        if word in word_dict: words.append(word_dict[word][0] / word_dict[word][1])
        elif '*' in word: words.append(-1)
        elif set(word) & set('1234567890'): words.append(-.5)
    while len(results) < 20: words.append(0)
    results.append(words)

clf = MLPClassifier()
train_dict = {'lyric': training}
training = pd.DataFrame(data = train_dict)
clf.fit(training, df_train['class'])
result_dict = {'lyric': results}
training = pd.DataFrame(data = train_dict)
pred = clf.predict(results)
df_test.insert(2, 'class', pred, True)
submission = df_test[['id', 'class']]
submission.to_csv("submission.csv", index=False)