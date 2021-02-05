# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:06:49 2021
@author: nannib
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('italian')
ps = nltk.PorterStemmer()
#ps = nltk.WordNetLemmatizer()

# here we clean text from punctuation, separate words in tokens (single words separated by comma)
# then we remove stopwords (or, is, and, etc.)
# finally we are using only stems for the words (is like lemma, the root of the word)
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    #text = [ps.lemmatize(word) for word in tokens if word not in stopwords] 
    return text

df = pd.read_csv('oot.csv', sep='\t', names=['text', 'author'], header=None)


#print(df.head())
X = df['text']
y = df['author']

# we can choose TF-IDF vectorization or CountVectorization
#vect = TfidfVectorizer(analyzer=clean_text, ngram_range=(1,2))
vect = CountVectorizer(analyzer=clean_text, ngram_range=(1,2))
X = vect.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#model = BernoulliNB()
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print( f'Train acc. {acc_train}, test acc. {acc_test}' )
testo=vect.transform(['oibò spacchiamo tutto!!'])
#print(testo)
print (model.predict(testo))
