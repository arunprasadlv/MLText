# # Homework with McDonald's sentiment data (Solution)

# ## Imaginary problem statement
# 
# McDonald's receives **thousands of customer comments** on their website per day, and many of them are negative. Their corporate employees don't have time to read every single comment, but they do want to read a subset of comments that they are most interested in. In particular, the media has recently portrayed their employees as being rude, and so they want to review comments about **rude service**.
# 
# McDonald's has hired you to develop a system that ranks each comment by the **likelihood that it is referring to rude service**. They will use your system to build a "rudeness dashboard" for their corporate employees, so that employees can spend a few minutes each day examining the **most relevant recent comments**.
# 
# ## Description of the data
# 
# Before hiring you, McDonald's used the [CrowdFlower platform](http://www.crowdflower.com/data-for-everyone) to pay humans to **hand-annotate** about 1500 comments with the **type of complaint**. The complaint types are listed below, with the encoding used in the data listed in parentheses:
# 
# - Bad Food (BadFood)
# - Bad Neighborhood (ScaryMcDs)
# - Cost (Cost)
# - Dirty Location (Filthy)
# - Missing Item (MissingFood)
# - Problem with Order (OrderProblem)
# - Rude Service (RudeService)
# - Slow Service (SlowService)
# - None of the above (na)

# for Python 2: use print only as a function
from __future__ import print_function


# ## Task 1
# 
# Read **`mcdonalds.csv`** into a pandas DataFrame and examine it. (It can be found in the **`data`** directory of the course repository.)
# 
# - The **policies_violated** column lists the type of complaint. If there is more than one type, the types are separated by newline characters.
# - The **policies_violated:confidence** column lists CrowdFlower's confidence in the judgments of its human annotators for that row (higher is better).
# - The **city** column is the McDonald's location.
# - The **review** column is the actual text comment.

# read mcdonalds.csv using a relative path
import pandas as pd
path = '../data/mcdonalds.csv'
mcd = pd.read_csv(path)


# examine the first three rows
mcd.head(3)


# examine the text of the first review
mcd.loc[0, 'review']


# ## Task 2
# 
# Remove any rows from the DataFrame in which the **policies_violated** column has a **null value**. Check the shape of the DataFrame before and after to confirm that you only removed about 50 rows.
# 
# - **Note:** Null values are also known as "missing values", and are encoded in pandas with the special value "NaN". This is distinct from the "na" encoding used by CrowdFlower to denote "None of the above". Rows that contain "na" should **not** be removed.
# - **Hint:** [How do I handle missing values in pandas?](https://www.youtube.com/watch?v=fCMrO_VzeL8&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=16) explains how to do this.

# examine the shape before removing any rows
mcd.shape


# count the number of null values in each column
mcd.isnull().sum()


# filter the DataFrame to only include rows in which policies_violated is not null
mcd = mcd[mcd.policies_violated.notnull()]

# alternatively, use the 'dropna' method to accomplish the same thing
mcd.dropna(subset=['policies_violated'], inplace=True)


# examine the shape after removing rows
mcd.shape


# ## Task 3
# 
# Add a new column to the DataFrame called **"rude"** that is 1 if the **policies_violated** column contains the text "RudeService", and 0 if the **policies_violated** column does not contain "RudeService". The "rude" column is going to be your response variable, so check how many zeros and ones it contains.
# 
# - **Hint:** [How do I use string methods in pandas?](https://www.youtube.com/watch?v=bofaC0IckHo&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=12) shows how to search for the presence of a substring, and [How do I change the data type of a pandas Series?](https://www.youtube.com/watch?v=V0AWyzVMf54&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=13) shows how to convert the boolean results (True/False) to integers (1/0).

# search for the string 'RudeService', and convert the boolean results to integers
mcd['rude'] = mcd.policies_violated.str.contains('RudeService').astype(int)


# confirm that it worked
mcd.loc[0:4, ['policies_violated', 'rude']]


# examine the class distribution
mcd.rude.value_counts()


# ## Task 4
# 
# 1. Define X (the **review** column) and y (the **rude** column).
# 2. Split X and y into training and testing sets (using the parameter **`random_state=1`**).
# 3. Use CountVectorizer (with the **default parameters**) to create document-term matrices from X_train and X_test.

# define X and y
X = mcd.review
y = mcd.rude


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# fit and transform X_train into X_train_dtm
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# transform X_test into X_test_dtm
X_test_dtm = vect.transform(X_test)
X_test_dtm.shape


# ## Task 5
# 
# Fit a Multinomial Naive Bayes model to the training set, calculate the **predicted probabilites** (not the class predictions) for the testing set, and then calculate the **AUC**. Repeat this task using a logistic regression model to see which of the two models achieves a better AUC.
# 
# - **Note:** Because McDonald's only cares about ranking the comments by the likelihood that they refer to rude service, **classification accuracy** is not the relevant evaluation metric. **Area Under the Curve (AUC)** is a more useful evaluation metric for this scenario, since it measures the ability of the classifier to assign higher predicted probabilities to positive instances than to negative instances.
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to calculate predicted probabilities and AUC, and my [blog post and video](http://www.dataschool.io/roc-curves-and-auc-explained/) explain AUC in-depth.

# import/instantiate/fit a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)


# calculate the predicted probability of rude=1 for each testing set observation
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]


# calculate the AUC
from sklearn import metrics
metrics.roc_auc_score(y_test, y_pred_prob)


# repeat this task using a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
metrics.roc_auc_score(y_test, y_pred_prob)


# ## Task 6
# 
# Using either Naive Bayes or logistic regression (whichever one had a better AUC in the previous step), try **tuning CountVectorizer** using some of the techniques we learned in class. Check the testing set **AUC** after each change, and find the set of parameters that increases AUC the most.
# 
# - **Hint:** It is highly recommended that you adapt the **`tokenize_test()`** function from class for this purpose, since it will allow you to iterate quickly through different sets of parameters.

# define a function that accepts a vectorizer and calculates the AUC
def tokenize_test(vect):
    
    # create document-term matrices using the vectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    # print the number of features that were generated
    print('Features:', X_train_dtm.shape[1])
    
    # use Multinomial Naive Bayes to calculate predicted probabilities
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
    
    # print the AUC
    print('AUC:', metrics.roc_auc_score(y_test, y_pred_prob))


# confirm that the AUC is identical to task 5 when using the default parameters
vect = CountVectorizer()
tokenize_test(vect)


# tune CountVectorizer to increase the AUC
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect)


# ## Task 7 (Challenge)
# 
# The **city** column might be predictive of the response, but we are not currently using it as a feature. Let's see whether we can increase the AUC by adding it to the model:
# 
# 1. Create a new DataFrame column, **review_city**, that concatenates the **review** text with the **city** text. One easy way to combine string columns in pandas is by using the [`Series.str.cat()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.cat.html) method. Make sure to use the **space character** as a separator, as well as replacing **null city values** with a reasonable string value (such as 'na').
# 2. Redefine X as the **review_city** column, and re-split X and y into training and testing sets.
# 3. When you run **`tokenize_test()`**, CountVectorizer will simply treat the city as an extra word in the review, and thus it will automatically be included in the model! Check to see whether it increased or decreased the AUC of your **best model**.

# concatenate review and city, separated by a space, replacing nulls with 'na'
mcd['review_city'] = mcd.review.str.cat(mcd.city, sep=' ', na_rep='na')


# examine review_city for the first row to confirm that it worked
mcd.loc[0, 'review_city']


# redefine X and y
X = mcd.review_city
y = mcd.rude


# re-split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# check whether it increased or decreased the AUC of my best model
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect)


# ## Task 8 (Challenge)
# 
# The **policies_violated:confidence** column may be useful, since it essentially represents a measurement of the training data quality. Let's see whether we can improve the AUC by only training the model using higher-quality rows!
# 
# To accomplish this, your first sub-task is to **calculate the mean confidence score for each row**, and then store those mean scores in a new column. For example, the confidence scores for the first row are `1.0\r\n0.6667\r\n0.6667`, so you should calculate a mean of `0.7778`. Here are the suggested steps:
# 
# 1. Using the [`Series.str.split()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.split.html) method, convert the **policies_violated:confidence** column into lists of one or more "confidence scores". Save the results as a new DataFrame column called **confidence_list**.
# 2. Define a function that calculates the mean of a list of numbers, and pass that function to the [`Series.apply()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html) method of the **confidence_list** column. That will calculate the mean confidence score for each row. Save those scores in a new DataFrame column called **confidence_mean**.
#     - **Hint:** [How do I apply a function to a pandas Series or DataFrame?](https://www.youtube.com/watch?v=P_q0tkYqvSk&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=30) explains how to use the `Series.apply()` method.

# examine the existing confidence column
mcd['policies_violated:confidence'].head()


# split the column into lists of one or more confidence scores
mcd['confidence_list'] = mcd['policies_violated:confidence'].str.split()
mcd.confidence_list.head()


import numpy as np

# define a function that accepts a list of strings and returns the mean
def mean_of_list(conf_list):
    
    # convert the list to a NumPy array of floats
    conf_array = np.array(conf_list, dtype=float)
    
    # return the mean of the array
    return np.mean(conf_array)


# calculate the mean confidence score for each row
mcd['confidence_mean'] = mcd.confidence_list.apply(mean_of_list)
mcd.confidence_mean.head()


# Your second sub-task is to **remove lower-quality rows from the training set**, and then repeat the model building and evaluation process. Here are the suggested steps:
# 
# 1. Remove all rows from X_train and y_train that have a **confidence_mean lower than 0.75**. Check their shapes before and after to confirm that you removed about 300 rows.
# 2. Use the **`tokenize_test()`** function to check whether filtering the training data increased or decreased the AUC of your **best model**.
#     - **Hint:** Even though X_train and y_train are separate from the mcd DataFrame, they can still be filtered using a boolean Series generated from mcd because all three objects share the same index.
#     - **Note:** It's important that we don't remove any rows from the testing set (X_test and y_test), because the testing set should be representative of the real-world data we will encounter in the future (which will contain both high-quality and low-quality rows).

# check the shapes of X_train and y_train before removing any rows
print(X_train.shape)
print(y_train.shape)


# remove any rows from X_train and y_train that have a confidence_mean lower than 0.75
X_train = X_train[mcd.confidence_mean >= 0.75]
y_train = y_train[mcd.confidence_mean >= 0.75]
print(X_train.shape)
print(y_train.shape)


# check whether it increased or decreased the AUC of my best model
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect)


# ## Task 9 (Challenge)
# 
# New comments have been submitted to the McDonald's website, and you need to **score them with the likelihood** that they are referring to rude service.
# 
# 1. Before making predictions on out-of-sample data, it is important to re-train your model on all relevant data using the tuning parameters and preprocessing steps that produced the best AUC above.
#     - In other words, X should be defined using either **all rows** or **only those rows with a confidence_mean of at least 0.75**, whichever produced a better AUC above.
#     - X should refer to either the **review column** or the **review_city column**, whichever produced a better AUC above.
#     - CountVectorizer should be instantiated with the **tuning parameters** that produced the best AUC above.
#     - **`train_test_split()`** should not be used during this process.
# 2. Build a document-term matrix (from X) called **X_dtm**, and examine its shape.
# 3. Read the new comments stored in **`mcdonalds_new.csv`** into a DataFrame called **new_comments**, and examine it.
# 4. If your model uses a **review_city** column, create that column in the new_comments DataFrame. (Otherwise, skip this step.)
# 5. Build a document_term matrix (from the **new_comments** DataFrame) called **new_dtm**, and examine its shape.
# 6. Train your best model (Naive Bayes or logistic regression) using **X_dtm** and **y**.
# 7. Predict the "rude probability" for each comment in **new_dtm**, and store the probabilities in an object called **new_pred_prob**.
# 8. Print the **full text** for each new comment alongside its **"rude probability"**. (You may need to [increase the max_colwidth](https://www.youtube.com/watch?v=yiO43TQ4xvc&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=28) to see the full text.) Examine the results, and comment on how well you think the model performed!

# define X and y using the data that produced the best AUC above
X = mcd.review_city
y = mcd.rude


# instantiate CountVectorizer with the tuning parameters that produced the best AUC above
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)


# fit and transform X into X_dtm
X_dtm = vect.fit_transform(X)
X_dtm.shape


# read mcdonalds_new.csv into new_comments
path = '../data/mcdonalds_new.csv'
new_comments = pd.read_csv(path)


# examine the new_comments DataFrame
new_comments


# concatenate review and city, separated by a space, replacing nulls with 'na'
new_comments['review_city'] = new_comments.review.str.cat(new_comments.city, sep=' ', na_rep='na')


# transform new_comments.review_city into new_dtm
new_dtm = vect.transform(new_comments.review_city)
new_dtm.shape


# train a MultinomialNB model
nb = MultinomialNB()
nb.fit(X_dtm, y)


# calculate the predicted probability of "rude service" for each new comment
new_pred_prob = nb.predict_proba(new_dtm)[:, 1]


# widen the column display
pd.set_option('display.max_colwidth', 1000)


# print the comment text alongside the predicted "rude probability"
# note: use sort() instead of sort_values() prior to pandas 0.17
pd.DataFrame({'comment':new_comments.review_city, 'rude_probability':new_pred_prob}).sort_values('rude_probability', ascending=False)


# reset the column display width
pd.reset_option('display.max_colwidth')


# **Comments:**
# 
# - The model correctly ranked all **"rude service" comments** (0, 2, 4, 7) above all other comments, thus accomplishing the overall objective.
# - Out of the three **positive comments** (1, 6, 9), the model properly ranked two of them the lowest.
# - The three comments which were **negative but not about rude service** (3, 5, 8) were ranked in the middle.
