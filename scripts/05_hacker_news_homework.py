# # Homework with Hacker News data
# 
# ## About Hacker News
# 
# [Hacker News](https://news.ycombinator.com/) is a popular "social news" website run by the startup incubator Y Combinator. It primarily includes news about technology, but also includes job postings and community-generated questions.
# 
# Any user can [submit](https://news.ycombinator.com/submit) a post to Hacker News. There are two types of posts: articles and discussions. To submit an **article**, the user includes a title and a URL. To submit a **discussion**, the user includes a title and additional text.
# 
# Users can upvote posts that they find interesting. Every post starts at 1 point, and each upvote adds an additional point. The most popular recent posts appear on the front page of Hacker News.
# 
# ## Description of the data
# 
# A [dataset of Hacker News posts](https://www.kaggle.com/hacker-news/hacker-news-posts) is hosted on Kaggle Datasets. It includes about one year of data, ending in September 2016. The following fields are included in the dataset:
# 
# - **title:** title of the post
# - **url:** URL of the post (if any)
# - **num_points:** number of points that the post received
# - **num_comments:** number of user comments on the post
# - **author:** name of the user that submitted the post
# - **created_at:** date and time the post was submitted
# 
# ## Problem statement
# 
# Your goal is to predict the likelihood that a post will be "popular", based on the data that is available at the time the post is submitted.
# 
# ## Tasks
# 
# 1. Get the data
# 2. Prepare the data
# 3. Explore the training data
# 4. Feature engineering
# 5. Define **`make_features()`**
# 6. Evaluate your model using cross-validation
# 7. Tune your model using grid search
# 8. Make predictions for the new data
# 9. Use text as the input data instead

# ## Task 1: Get the data
# 
# 1. Go to the [Kaggle Datasets](https://www.kaggle.com/hacker-news/hacker-news-posts) page, and click the **Download** button.
# 2. Unzip **`hacker-news-posts.zip`**, and then move **`HN_posts_year_to_Sep_26_2016.csv`** to a directory where you can easily access it.
# 3. Read the file into a pandas DataFrame called **"hn"**.
# 4. Either during or after the file reading process, convert the **created_at** column to datetime format.
# 
#     - **Hint:** [How do I work with dates and times in pandas?](https://www.youtube.com/watch?v=yCgJGsg0Xa4&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=25) explains how to do this.

# ## Task 2: Prepare the data
# 
# 1. Create a new column called **"popular"** that contains **1** if the post received more than 5 points, and **0** otherwise. This will be the response variable that you are trying to predict.
# 2. Split the **hn** DataFrame into two separate DataFrames. The first DataFrame should be called **"train"**, and should contain all posts before July 1, 2016. The second DataFrame should be called **"new"**, and should contain the remaining posts.
# 
#     - **Hint:** [How do I work with dates and times in pandas?](https://www.youtube.com/watch?v=yCgJGsg0Xa4&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=25) explains how to do this. Make sure that all rows from **hn** are in either **train** or **new**, but not both.
#     - **Hint:** When you are creating **train** and **new**, you should use the [`DataFrame.copy()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.copy.html) method to make sure that you are creating separate objects (rather than references to the **hn** DataFrame).
#     - **Note:** You will be building a model using the posts in **train** and making predictions for the posts in **new** (our simulated future data).

# ## Task 3: Explore the training data
# 
# Explore the **train** DataFrame to gain an understanding of the dataset.
# 
# **Note:** At no time should you explore the **new** DataFrame, since this is our simulated future data which you would not have access to in the "real world".

# ## Task 4: Feature engineering
# 
# Create new features in **train** that you think might be relevant to predicting the response, **popular**. After creating each feature, check whether it is likely to be a useful feature.
# 
# For this task, don't use **`CountVectorizer`**. (You will use it in Task 9 instead.)
# 
# **Note:** Think very carefully about which features you would be "allowed" to use in the real world. If a feature incorporates future data that would not be available **at the time of post submission**, then it can't be used in your model.

# ## Task 5: Define **`make_features()`**
# 
# 1. Define a function, **`make_features()`**, that accepts a DataFrame and returns a DataFrame with your engineered features added. You should only include features that you think might be useful for predicting popularity.
# 2. Re-split the **hn** DataFrame into **train** and **new** (using the code from Task 2) to return them to their original contents.
# 3. Run **`make_features()`** on **train** and **new**, and check that your features were successfully created.

# ## Task 6: Evaluate your model using cross-validation
# 
# 1. Define **X** and **y** using your chosen feature columns from **train**.
# 2. Choose a classification model, and then use **`cross_val_score`** to evaluate your model. We're going to use AUC as the evaluation metric, so use the parameter **`scoring='roc_auc'`**.
# 3. **Optional:** Try adding features to your model that would not be "allowed" in the real world (because they incorporate information about the future), and see how that affects your AUC. (Be sure to remove these features from your model before moving on to the next task!)
# 
#     - **Note:** An AUC of 1.0 represents a perfect model, and an AUC of 0.5 represents random guessing. You can think of 0.5 as the AUC of the "null model". (My [blog post and video](http://www.dataschool.io/roc-curves-and-auc-explained/) explain AUC in more depth.)

# ## Task 7: Tune your model using grid search
# 
# Use **`GridSearchCV`** to find the optimal tuning parameters for your model.

# ## Task 8: Make predictions for the new data
# 
# 1. Create a DataFrame called **"X_new"** that includes the same feature columns you used to train your model.
# 2. Train your best model (found during grid search) using **X** and **y**.
# 3. Calculate the predicted probability of popularity for all posts in **X_new**.
# 4. Calculate the AUC of your model by evaluating your predicted probabilities against the **popular** column in the **new** DataFrame. (It should be similar to the AUC that was output by **`GridSearchCV`**.)
# 
#     - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to calculate predicted probabilities and AUC.

# ## Task 9: Use text as the input data instead
# 
# 1. Define a new **X** using the **title** column from **train**.
# 2. Create a **`Pipeline`** of **`CountVectorizer`** and the model of your choice.
# 3. Use **`cross_val_score`** to properly evaluate the AUC of your pipeline.
# 4. **Optional:** See if you can increase the AUC by changing what you use as the input text.
# 5. Train the pipeline on **X** and **y**, calculate predicted probabilities for all posts in the **new** DataFrame, and calculate the AUC.
