# # Working a Text-Based Data Science Problem

# ## Agenda
# 
# 1. Reading in and exploring the data
# 2. Feature engineering
# 3. Model evaluation using **`train_test_split`** and **`cross_val_score`**
# 4. Making predictions for new data
# 5. Searching for optimal tuning parameters using **`GridSearchCV`**
# 6. Extracting features from text using **`CountVectorizer`**
# 7. Chaining steps into a **`Pipeline`**

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Reading in and exploring the data

import pandas as pd
train = pd.read_json('../data/train.json')
train.head()


train.shape


# count the number of null values in each column
train.isnull().sum()


train.dtypes


# select row 0, column 'ingredients'
train.loc[0, 'ingredients']


# ingredients are stored as a list of strings, not as a string
type(train.loc[0, 'ingredients'])


# examine the class distribution
train.cuisine.value_counts()


# ## Part 2: Feature engineering
# 
# - Feature engineering is the process through which you create features that **don't natively exist** in the dataset.
# - Your goal is to create features that contain the **signal** from the data (with respect to the response value), rather than the **noise**.

# ### Example: Number of ingredients

# count the number of ingredients in each recipe
train['num_ingredients'] = train.ingredients.apply(len)
train.head()


# for each cuisine, calculate the mean number of ingredients
train.groupby('cuisine').num_ingredients.mean()


# for each cuisine, "describe" the number of ingredients (and unstack into a DataFrame)
train.groupby('cuisine').num_ingredients.describe().unstack()


# box plot of number ingredients for each cuisine
train.boxplot('num_ingredients', by='cuisine')


# ### Example: Mean length of ingredient names

sample_recipe = train.loc[3, 'ingredients']
print(sample_recipe)


import numpy as np


# define a function that calculates the mean string length from a list of strings
def mean_string_length(list_of_strings):
    return np.mean([len(string) for string in list_of_strings])


mean_string_length(sample_recipe)


# calculate the mean ingredient length for each recipe (two different ways)
train['ingredient_length'] = train.ingredients.apply(mean_string_length)
train['ingredient_length'] = train.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
train.head()


# box plot of mean ingredient length for each cuisine
train.boxplot('ingredient_length', by='cuisine')


# define a function that accepts a DataFrame and adds new features
def make_features(df):
    df['num_ingredients'] = df.ingredients.apply(len)
    df['ingredient_length'] = df.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
    return df


# check that the function works
train = make_features(pd.read_json('../data/train.json'))
train.head()


# ## Part 3: Model evaluation using `train_test_split` and `cross_val_score`
# 
# - The motivation for model evaluation is that you need a way to **choose between models** (different model types, tuning parameters, and features).
# - You use a model evaluation procedure to estimate how well a model will **generalize** to out-of-sample data.
# - This requires a model evaluation metric to **quantify** a model's performance.

# define X and y
feature_cols = ['num_ingredients', 'ingredient_length']
X = train[feature_cols]
y = train.cuisine


print(X.shape)
print(y.shape)


# note: response values are strings (not numbers)
y.values


# [KNeighborsClassifier documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

# use KNN with K=100
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)


# ### Train/test split

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# make class predictions for the testing set
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)


# check the classification accuracy of KNN's predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# ### K-fold cross-validation
# 
# - Train/test split is **faster and more flexible**
# - Cross-validation provides a **more accurate** estimate of out-of-sample performance

# evaluate with 5-fold cross-validation (using X instead of X_train)
from sklearn.cross_validation import cross_val_score
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()


# ### Null model
# 
# - For **classification** problems, the null model always predicts the most frequent class from the training data.
# - For **regression problems**, the null model always predicts the mean of the response value from the training data.
# - It can be a useful **baseline model** against which your model is measured.

# calculate the null accuracy
y_test.value_counts().head(1) / y_test.shape


# [DummyClassifier documentation](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

# use DummyClassifier instead
from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy='most_frequent')
dumb.fit(X_train, y_train)
y_pred_class = dumb.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# ## Part 4: Making predictions for new data

# read in test.json and add the additional features
new = make_features(pd.read_json('../data/test.json'))
new.head()


new.shape


# create a DataFrame of the relevant columns from the new data
X_new = new[feature_cols]
X_new.head()


X_new.shape


# train KNN on ALL of the training data (using X instead of X_train)
knn.fit(X, y)


# make class predictions for the new data
new_pred_class_knn = knn.predict(X_new)
new_pred_class_knn


new_pred_class_knn.shape


# create a DataFrame that only contains the IDs and predicted classes for the new data
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class_knn}).set_index('id').head()


# create a submission file from that DataFrame (score: 0.21742)
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class_knn}).set_index('id').to_csv('sub1.csv')


# ## Part 5: Searching for optimal tuning parameters using `GridSearchCV`

# reminder of the cross-validated accuracy of KNN with K=100
knn = KNeighborsClassifier(n_neighbors=100)
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()


# [GridSearchCV documentation](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html)

from sklearn.grid_search import GridSearchCV


# define a "parameter grid" in which the key is the parameter and the value is a list of options to try
param_grid = {}
param_grid['n_neighbors'] = [100, 200]
param_grid


# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')


# run the grid search
grid.fit(X, y)


# examine the scores for each parameter option
grid.grid_scores_


# try K=200 to 1000 (by 200)
param_grid = {}
param_grid['n_neighbors'] = list(range(200, 1001, 200))
param_grid


grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')


# run the grid search
grid.fit(X, y)


# examine the scores for each parameter option
grid.grid_scores_


# extract only the mean scores
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
grid_mean_scores


# line plot of K value (x-axis) versus accuracy (y-axis)
import matplotlib.pyplot as plt
plt.plot(list(range(200, 1001, 200)), grid_mean_scores)


# print the single best score and parameters that produced that score
print(grid.best_score_)
print(grid.best_params_)


# ## Part 6: Extracting features from text using `CountVectorizer`

# reminder: ingredients are stored as a list of strings, not as a string
train.loc[0, 'ingredients']


# convert each list of ingredients into a string
train.ingredients.astype(str)[0]


# update make_features to create a new column 'ingredients_str'
def make_features(df):
    df['num_ingredients'] = df.ingredients.apply(len)
    df['ingredient_length'] = df.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
    df['ingredients_str'] = df.ingredients.astype(str)
    return df


# run make_features and check that it worked
train = make_features(pd.read_json('../data/train.json'))
train.loc[0, 'ingredients_str']


# define X and y
X = train.ingredients_str
y = train.cuisine


# import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect


# create a document-term matrix from all of the training data
X_dtm = vect.fit_transform(X)
X_dtm.shape


# examine the features that were created
print(vect.get_feature_names()[0:100])


# replace the regex pattern that is used for tokenization
vect = CountVectorizer(token_pattern=r"'([a-z ]+)'")
X_dtm = vect.fit_transform(X)
X_dtm.shape


# examine the features that were created
print(vect.get_feature_names()[0:100])


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# slightly improper cross-validation
cross_val_score(nb, X_dtm, y, cv=5, scoring='accuracy').mean()


# **Why is this improper cross-validation?**
# 
# - Normally, we split the data into training and testing sets **before** creating the document-term matrix. But since **`cross_val_score`** does the splitting for you, we passed it the feature matrix (**`X_dtm`**) rather than the raw text (**`X`**).
# - However, that does not appropriately simulate the real world, in which your out-of-sample data will contain **features that were not seen** during model training.
# 
# **What's the solution?**
# 
# - We need a way to pass **`X`** (not **`X_dtm`**) to **`cross_val_score`**, and have the feature creation (via **`CountVectorizer`**) occur **within each fold** of cross-validation.
# - We will do this by using a **`Pipeline`**.

# ## Part 7: Chaining steps into a `Pipeline`

# examine the numeric columns of the training data
train.describe()


# [Imputer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)

# define '1' as a missing value and impute a replacement using the median
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=1, strategy='median')


# [make_pipeline documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)

# create a pipeline of missing value imputation and KNN
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(imp, knn)


# examine the pipeline steps
pipe.steps


# [Pipeline documentation](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

# alternative method for creating the identical pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([('imputer', imp), ('kneighborsclassifier', knn)])


# fit and predict using the entire pipeline
pipe.fit(X_train, y_train)
y_pred_class = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred_class)


# ### Using a `Pipeline` for proper cross-validation

# create a pipeline of vectorization and Naive Bayes
pipe = make_pipeline(vect, nb)
pipe.steps


# We can now pass **`X`** (instead of **`X_dtm`**) to **`cross_val_score`**, and the vectorization will occur **within each fold** of cross-validation.

# X is just a Series of strings
X.head()


# cross-validate the entire pipeline
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# ### Making predictions using a `Pipeline`

# fit the pipeline (rather than just the model)
pipe.fit(X, y)


# read in test.json and add the additional features
new = make_features(pd.read_json('../data/test.json'))


# define X_new as a Series of strings
X_new = new.ingredients_str


# use the pipeline to make predictions for the new data
new_pred_class_pipe = pipe.predict(X_new)


# create a submission file (score: 0.73663)
pd.DataFrame({'id':new.id, 'cuisine':new_pred_class_pipe}).set_index('id').to_csv('sub2.csv')

