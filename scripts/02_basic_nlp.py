# # Basic Natural Language Processing (NLP)

# ## What is NLP?
# 
# - Using computers to process (analyze, understand, generate) natural human languages
# - Most knowledge created by humans is unstructured text, and we need a way to make sense of it
# - Build probabilistic model using data about a language
# - Requires an understanding of language and the world

# ## Higher level "task areas"
# 
# - **Information retrieval**: Find relevant results and similar results
#     - [Google](https://www.google.com/)
# - **Information extraction**: Structured information from unstructured documents
#     - [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en)
# - **Machine translation**: One language to another
#     - [Google Translate](https://translate.google.com/)
# - **Text simplification**: Preserve the meaning of text, but simplify the grammar and vocabulary
#     - [Rewordify](https://rewordify.com/)
#     - [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
# - **Predictive text input**: Faster or easier typing
#     - [My application](https://justmarkham.shinyapps.io/textprediction/)
#     - [A much better application](https://farsite.shinyapps.io/swiftkey-cap/)
# - **Sentiment analysis**: Attitude of speaker
#     - [Hater News](http://haternews.herokuapp.com/)
# - **Automatic summarization**: Extractive or abstractive summarization
#     - [autotldr](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)
# - **Natural Language Generation**: Generate text from data
#     - [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052)
#     - [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763)
# - **Speech recognition and generation**: Speech-to-text, text-to-speech
#     - [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html)
#     - [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo)
# - **Question answering**: Determine the intent of the question, match query with knowledge base, evaluate hypotheses
#     - [How did supercomputer Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/)
#     - [IBM's Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html)
#     - [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

# ## Lower level "components"
# 
# - **Tokenization**: breaking text into tokens (words, sentences, n-grams)
# - **Stop word removal**: removing common words
# - **TF-IDF**: computing word importance
# - **Stemming and lemmatization**: reducing words to their base form
# - **Part-of-speech tagging**
# - **Named entity recognition**: person/organization/location
# - **Segmentation**: "New York City subway"
# - **Word sense disambiguation**: "buy a mouse"
# - **Spelling correction**
# - **Language detection**
# - **Machine learning**

# ## Agenda
# 
# 1. Reading in the Yelp reviews corpus
# 2. Tokenizing the text
# 3. Comparing the accuracy of different approaches
# 4. Removing frequent terms (stop words)
# 5. Removing infrequent terms
# 6. Handling Unicode errors

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Reading in the Yelp reviews corpus

# - "corpus" = collection of documents
# - "corpora" = plural form of corpus

# read yelp.csv into a DataFrame using a relative path
import pandas as pd
path = '../data/yelp.csv'
yelp = pd.read_csv(path)


# alternative: read from a URL instead
# path = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
# yelp = pd.read_csv(path)


# examine the first three rows
yelp.head(3)


# examine the text for the first row
yelp.loc[0, 'text']


# **Goal:** Distinguish between 5-star and 1-star reviews using **only** the review text. (We will not be using the other columns.)

# examine the class distribution
yelp.stars.value_counts().sort_index()


# create a new DataFrame that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]


# examine the shape
yelp_best_worst.shape


# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# examine the object shapes
print(X_train.shape)
print(X_test.shape)


# ## Part 2: Tokenizing the text

# - **What:** Separate text into units such as words, n-grams, or sentences
# - **Why:** Gives structure to previously unstructured text
# - **Notes:** Relatively easy with English language text, not easy with some languages

# use CountVectorizer to create document-term matrices from X_train and X_test
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# fit and transform X_train
X_train_dtm = vect.fit_transform(X_train)


# only transform X_test
X_test_dtm = vect.transform(X_test)


# examine the shapes: rows are documents, columns are terms (aka "tokens" or "features")
print(X_train_dtm.shape)
print(X_test_dtm.shape)


# examine the last 50 features
print(vect.get_feature_names()[-50:])


# show default parameters for CountVectorizer
vect


# [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# - **lowercase:** boolean, True by default
#     - Convert all characters to lowercase before tokenizing.

# don't convert to lowercase
vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# - **ngram_range:** tuple (min_n, max_n), default=(1, 1)
#     - The lower and upper boundary of the range of n-values for different n-grams to be extracted.
#     - All values of n such that min_n <= n <= max_n will be used.

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# examine the last 50 features
print(vect.get_feature_names()[-50:])


# ## Part 3: Comparing the accuracy of different approaches

# **Approach 1:** Always predict the most frequent class

# calculate null accuracy
y_test.value_counts().head(1) / y_test.shape


# **Approach 2:** Use the default parameters for CountVectorizer

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


# use the default parameters
vect = CountVectorizer()
tokenize_test(vect)


# **Approach 3:** Don't convert to lowercase

# don't convert to lowercase
vect = CountVectorizer(lowercase=False)
tokenize_test(vect)


# **Approach 4:** Include 1-grams and 2-grams

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)


# **Summary:** Tuning CountVectorizer is a form of **feature engineering**, the process through which you create features that don't natively exist in the dataset. Your goal is to create features that contain the **signal** from the data (with respect to the response value), rather than the **noise**.

# ## Part 4: Removing frequent terms (stop words)

# - **What:** Remove common words that appear in most documents
# - **Why:** They probably don't tell you much about your text

# show vectorizer parameters
vect


# - **stop_words:** string {'english'}, list, or None (default)
#     - If 'english', a built-in stop word list for English is used.
#     - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
#     - If None, no stop words will be used.

# remove English stop words
vect = CountVectorizer(stop_words='english')
tokenize_test(vect)


# examine the stop words
print(sorted(vect.get_stop_words()))


# - **max_df:** float in range [0.0, 1.0] or int, default=1.0
#     - When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.

# ignore terms that appear in more than 50% of the documents
vect = CountVectorizer(max_df=0.5)
tokenize_test(vect)


# - **stop\_words\_:** Terms that were ignored because they either:
#     - occurred in too many documents (max_df)
#     - occurred in too few documents (min_df)
#     - were cut off by feature selection (max_features)

# examine the terms that were removed due to max_df ("corpus-specific stop words")
print(vect.stop_words_)


# vect.stop_words_ is completely distinct from vect.get_stop_words()
print(vect.get_stop_words())


# ## Part 5: Removing infrequent terms

# - **max_features:** int or None, default=None
#     - If not None, build a vocabulary that only considers the top max_features ordered by term frequency across the corpus.

# only keep the top 1000 most frequent terms
vect = CountVectorizer(max_features=1000)
tokenize_test(vect)


# - **min_df:** float in range [0.0, 1.0] or int, default=1
#     - When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold. (This value is also called "cut-off" in the literature.)
#     - If float, the parameter represents a proportion of documents.
#     - If integer, the parameter represents an absolute count.

# only keep terms that appear in at least 2 documents
vect = CountVectorizer(min_df=2)
tokenize_test(vect)


# include 1-grams and 2-grams, and only keep terms that appear in at least 2 documents
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
tokenize_test(vect)


# **Guidelines for tuning CountVectorizer:**
# 
# - Use your knowledge of the **problem** and the **text**, and your understanding of the **tuning parameters**, to help you decide what parameters to tune and how to tune them.
# - **Experiment**, and let the data tell you the best approach!

# ## Part 6: Handling Unicode errors

# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#decoding-text-files):
# 
# > Text is made of **characters**, but files are made of **bytes**. These bytes represent characters according to some **encoding**. To work with text files in Python, their bytes must be decoded to a character set called **Unicode**. Common encodings are ASCII, Latin-1 (Western Europe), KOI8-R (Russian) and the universal encodings UTF-8 and UTF-16. Many others exist.
# 
# **Why should you care?**
# 
# When working with text in Python, you are likely to encounter errors related to encoding, and understanding Unicode will help you to troubleshoot these errors.
# 
# **Unicode basics:**
# 
# - Unicode is a system that assigns a unique number for every character in every language. These numbers are called **code points**. For example, the [code point](http://www.unicode.org/charts/index.html) for "A" is U+0041, and the official name is "LATIN CAPITAL LETTER A".
# - An **encoding** specifies how to store the code points in memory:
#     - **UTF-8** is the most popular Unicode encoding. It uses 8 to 32 bits to store each character.
#     - **UTF-16** is the second most popular Unicode encoding. It uses 16 or 32 bits to store each character.
#     - **UTF-32** is the least popular Unicode encoding. It uses 32 bits to store each character.
# 
# **ASCII basics:**
# - ASCII is an encoding from the 1960's that uses 8 bits to store each character, and only supports **English characters**.
# - ASCII-encoded files are sometimes called **plain text**.
# - UTF-8 is **backward-compatible** with ASCII, because the first 8 bits of a UTF-8 encoding are identical to the ASCII encoding.
# 
# The default encoding in **Python 2** is ASCII. The default encoding in **Python 3** is UTF-8.

# Python 2: examine two types of strings
print(type('hello'))
print(type(u'hello'))


# Python 2: 'decode' converts 'str' to 'unicode'
'hello'.decode(encoding='utf-8')


# Python 2: 'encode' converts 'unicode' to 'str'
u'hello'.encode(encoding='utf-8')


# Python 3: examine two types of strings
# print(type(b'hello'))
# print(type('hello'))


# Python 3: 'decode' converts 'bytes' to 'str'
# b'hello'.decode(encoding='utf-8')


# Python 3: 'encode' converts 'str' to 'bytes'
# 'hello'.encode(encoding='utf-8')


# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#decoding-text-files):
# 
# > The text feature extractors in scikit-learn know how to **decode text files**, but only if you tell them what encoding the files are in. The CountVectorizer takes an **encoding parameter** for this purpose. For modern text files, the correct encoding is probably **UTF-8**, which is therefore the default (encoding="utf-8").
# 
# > If the text you are loading is not actually encoded with UTF-8, however, you will get a **UnicodeDecodeError**. The vectorizers can be told to be silent about decoding errors by setting the **decode_error parameter** to either "ignore" or "replace".

# read in a single Yelp review
path = '../data/yelp_single.csv'
single = pd.read_csv(path)
single.head()


# instantiate CountVectorizer with the default parameters
vect = CountVectorizer()
vect


# this will generate a UnicodeDecodeError
# vect.fit(single.text)


# method 1 to fix the error: ignore errors
vect = CountVectorizer(decode_error='ignore')
vect.fit(single.text)
features1 = vect.get_feature_names()
print(features1)


# method 2 to fix the error: guess the encoding
vect = CountVectorizer(encoding='latin-1')
vect.fit(single.text)
features2 = vect.get_feature_names()
print(features2)


# what "unique" features were created by method 1?
set(features1) - set(features2)


# what "unique" features were created by method 2?
set(features2) - set(features1)


# \x escape sequence means the next two characters should be interpreted as hexadecimal digits
print(u'cevich\xe9')


# use the print method on all three features at once
print(*set(features2) - set(features1))

