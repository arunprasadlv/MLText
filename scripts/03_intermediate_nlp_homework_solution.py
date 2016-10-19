# # Intermediate NLP Homework (Solution)
# 
# Using TF-IDF, write a document summarizer for a corpus of your choosing, but summarize using full sentences or paragraphs rather than individual words.

# for Python 2: use print only as a function
from __future__ import print_function


# read yelp.csv into a DataFrame using a relative path
import pandas as pd
path = '../data/yelp.csv'
yelp = pd.read_csv(path)


# create a document-term matrix using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
dtm = vect.fit_transform(yelp.text)
dtm.shape


# create a list of all of the features
features = vect.get_feature_names()
len(features)


import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

def summarize():
    
    # choose a random review that has at least 10 sentences
    num_sentences = 0
    while num_sentences < 10:
        review_id = np.random.randint(0, yelp.shape[0])
        review_text = yelp.loc[review_id, 'text']
        review_blob = TextBlob(review_text)
        num_sentences = len(review_blob.sentences)
    
    # create a list of all unique words in the review using CountVectorizer
    vect = CountVectorizer()
    vect.fit([review_text])
    unique_words = vect.get_feature_names()
    
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in unique_words:
        word_scores[word] = dtm[review_id, features.index(word)]
    
    # calculate the mean TF-IDF score for each sentence that has at least 6 words
    sentences = review_blob.sentences
    sentence_scores = []
    for sentence in sentences:
        sentence_words = sentence.words.lower()
        if len(sentence_words) >= 6:
            sentence_score = np.mean([word_scores[word] for word in sentence_words if word in unique_words])
            sentence_scores.append((sentence_score, sentence))
    
    # print sentences with the top 3 TF-IDF scores
    print('TOP SCORING SENTENCES:')
    top_scores = sorted(sentence_scores, reverse=True)[0:3]
    for score, sentence in top_scores:
        print(sentence)
    
    # print 3 random sentences (for comparison)
    print('\n' + 'RANDOM SENTENCES:')
    random_sentences = np.random.choice(sentences, size=3, replace=False)
    for sentence in random_sentences:
        print(sentence)
    
    # print the review
    print('\n' + 'REVIEW:' + '\n' + review_text)


summarize()
