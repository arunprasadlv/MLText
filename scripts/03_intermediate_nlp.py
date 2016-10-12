# # Intermediate Natural Language Processing (NLP)

# ## Agenda
# 
# 1. Reading in the Yelp reviews corpus
# 2. Term Frequency-Inverse Document Frequency (TF-IDF)
# 3. Using TF-IDF to summarize a Yelp review
# 4. Sentiment analysis using TextBlob

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Reading in the Yelp reviews corpus

# read yelp.csv into a DataFrame using a relative path
import pandas as pd
path = '../data/yelp.csv'
yelp = pd.read_csv(path)


# alternative: read from a URL instead
# path = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
# yelp = pd.read_csv(path)


# examine the first row
yelp.head(1)


# create a new DataFrame that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]
yelp_best_worst.shape


# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    
    # create document-term matrices using the vectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    # print the number of features that were generated
    print('Features: ', X_train_dtm.shape[1])
    
    # use Multinomial Naive Bayes to predict the star rating
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    
    # print the accuracy of its predictions
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))


# ## Part 2: Term Frequency-Inverse Document Frequency (TF-IDF)

# - **What:** Compute the "relative frequency" that a word appears in a document compared to its frequency across all documents in the corpus
# - **Why:** More useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents)
# - **Notes:** Useful for search engine relevance scoring, text summarization, document clustering, and classification

# example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# ### Calculating Term Frequency (TF):

# Term Frequency (count the number of occurrences of each term in each document)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf


# ### Calculating Document Frequency (DF):

# examine default parameters of CountVectorizer
vect


# [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
# 
# - **binary:** boolean, default=False
#     - If True, all non-zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.

# use 'binary=True' parameter to check for occurrence (1) or non-occurrence (0)
vect = CountVectorizer(binary=True)
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())


# Document Frequency (count the number of documents in which each term appears)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())


# ### Calculating TF-IDF from TF and DF:

# TF-IDF (simple version)
tf/df


# [TfidfVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

# TfidfVectorizer (CountVectorizer + TfidfTransformer)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
vect


# TF-IDF (scikit-learn's default implementation)
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())


# ### Using TF-IDF for classification:

# use CountVectorizer with default parameters
vect = CountVectorizer()
tokenize_test(vect)


# use TfidfVectorizer with default parameters
vect = TfidfVectorizer()
tokenize_test(vect)


# tune TfidfVectorizer for better results
vect = TfidfVectorizer(norm=None)
tokenize_test(vect)


# make TfidfVectorizer behave like CountVectorizer
vect = TfidfVectorizer(norm=None, use_idf=False)
tokenize_test(vect)


# ## Part 3: Using TF-IDF to summarize a Yelp review

# Reddit's [autotldr](https://np.reddit.com/r/autotldr/comments/31b9fm/faq_autotldr_bot/) uses the [SMMRY](http://smmry.com/about) algorithm, which is based on TF-IDF!

# create a document-term matrix (minus stop words) using TfidfVectorizer
vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(yelp.text)
dtm.shape


# create a list of all of the features
features = vect.get_feature_names()
len(features)


import numpy as np

def summarize():
    
    # choose a random review that is at least 400 characters
    review_length = 0
    while review_length < 400:
        review_id = np.random.randint(0, yelp.shape[0])
        review_text = yelp.loc[review_id, 'text']
        review_length = len(review_text)
    
    # create a list of all unique words in the review (minus stop words) using CountVectorizer
    vect = CountVectorizer(stop_words='english')
    vect.fit([review_text])
    unique_words = vect.get_feature_names()
    
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in unique_words:
        word_scores[word] = dtm[review_id, features.index(word)]
    
    # print words with the top 5 TF-IDF scores
    print('TOP SCORING WORDS:')
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[0:5]
    for word, score in top_scores:
        print(word)
    
    # print 5 random words (for comparison)
    print('\n' + 'RANDOM WORDS:')
    random_words = np.random.choice(list(word_scores.keys()), size=5, replace=False)
    for word in random_words:
        print(word)
    
    # print the review
    print('\n' + review_text)


# summarize a random Yelp review
summarize()


# ## Part 4: Sentiment analysis using TextBlob

# [TextBlob](https://textblob.readthedocs.org): "Simplified Text Processing"

# print the first review
print(yelp_best_worst.loc[0, 'text'])


# save it as a TextBlob object
from textblob import TextBlob
review = TextBlob(yelp_best_worst.loc[0, 'text'])


# examine the words
review.words


# examine the sentences
review.sentences


# Sentiment is the "attitude" of the speaker:
# 
# - **polarity** ranges from -1 (most negative) to 1 (most positive)
# - **subjectivity** ranges from 0 (very objective) to 1 (very subjective)

# examine the review sentiment
print(review.sentiment.polarity)
print(review.sentiment.subjectivity)


# quick aside: understanding the apply method
yelp['length'] = yelp.text.apply(len)
yelp.head(2)


# define a function that accepts text and returns the polarity
def detect_sentiment(text):
    
    # use this line for Python 2 (avoids UnicodeDecodeError for some reviews)
    blob = TextBlob(text.decode(encoding='utf-8'))
    
    # use this line instead for Python 3
    # blob = TextBlob(text)
    
    # return the polarity
    return blob.sentiment.polarity


# create a new DataFrame column for sentiment (WARNING: SLOW!)
yelp['sentiment'] = yelp.text.apply(detect_sentiment)
yelp.head(2)


# box plot of sentiment grouped by stars
yelp.boxplot('sentiment', by='stars')


# reviews with most positive sentiment
yelp.loc[yelp.sentiment == 1, 'text'].head()


# reviews with most negative sentiment
yelp.loc[yelp.sentiment == -1, 'text'].head()


# widen the column display
pd.set_option('display.max_colwidth', 500)


# negative sentiment in a 5-star review
yelp.loc[(yelp.stars == 5) & (yelp.sentiment < -0.3), 'text'].head(1)


# positive sentiment in a 1-star review
yelp.loc[(yelp.stars == 1) & (yelp.sentiment > 0.5), 'text'].head(1)


# reset the column display width
pd.reset_option('display.max_colwidth')

