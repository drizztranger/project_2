
# coding: utf-8

# In[ ]:


# The objective of this file is to extract feature and show example of using it to machine learning
import numpy as np
all_Y_train=np.r_[np.zeros((12500,1))+1,np.zeros((12500,1))] # Get output of training data
import pandas as pd
all_X_train = pd.read_pickle('all_X_train.pkl') #load training data frame
all_X_test = pd.read_pickle('all_X_test.pkl') #load test data frame
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(all_X_train[0], all_Y_train, train_size=0.8, test_size=0.2) # slit training and validation set

# Extraction of data
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import string

def tokenize(text): #integrate stem into TfidfVectorizer
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


Vectorizer = TfidfVectorizer(max_features=5000,tokenizer=tokenize,stop_words='english',token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',ngram_range=(1,3),binary=True,min_df=0,analyzer='word').fit(X_train)
# max_features : number of features wanted, if want to use all of features, just delete it
# tokenizer: if or not use stemmer. If do not need it, just delete it
# ngram_range: number of grams considered
# binary: The outcome of this Vectorizer is binary. If it is False, the the outcome is the count of word occurance
# min_df: If frequency < min_df, then these feature would be deleted.
X_train_tfidf = Vectorizer.transform(X_train) # Get feature of training set
X_valid_tfidf = Vectorizer.transform(X_valid) # Get feature of validation set
X_test_tfidf = Vectorizer.transform(all_X_test[0])# Get feature of test set



# Training model
clf = BernoulliNB().fit(X_train_tfidf, y_train) # Training BernoulliNB model
y_pred = clf.predict(X_valid_tfidf) # Predict validation set
print(metrics.classification_report(y_valid, y_pred,
    )) # Print the accuracy of prediction based on validation set

all_y_pred = clf.predict(X_test_tfidf) # Get prediction of test set
np.savetxt('new.csv', all_y_pred, delimiter = ',') # Save prediction result into csc file

