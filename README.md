## Machine Learning with Text in Python

* **Instructor:** [Kevin Markham](http://www.dataschool.io/about/)
* **Teaching Assistant:** [Alex Egorenkov](https://www.linkedin.com/in/aegorenkov)
* [Course information and FAQs](http://www.dataschool.io/learn/)
* [Slack team](https://mltextpython.slack.com/)

### Course Schedule

The course begins on September 28 and ends on November 22.

* [Before the Course](#before-the-course)
* [Week 1: Working with Text Data in scikit-learn](#week-1-working-with-text-data-in-scikit-learn)
* [Week 2: Basic Natural Language Processing (NLP)](#week-2-basic-natural-language-processing-nlp)
* Week 3: Intermediate NLP and Basic Regular Expressions (begins October 12)
* Week 4: Intermediate Regular Expressions (begins October 19)
* Week 5: Working a Text-Based Data Science Problem (begins October 26)
* Week 6: Advanced Machine Learning Techniques (begins November 2)
* Week 7: Private Kaggle competition (begins November 9)
* Week 8: Private Kaggle competition (begins November 16)
* After the Course

### Course Videos

Links to the video recordings can be found under each section below. Alternatively, you can [view all of the videos on one page](http://www.dataschool.io/mltext3-videos/), or [view the YouTube playlist](https://www.youtube.com/playlist?list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj).

### Live Webcasts

Live webcasts will take place every Sunday at 8:00pm ET and every Tuesday at 1:00pm ET. During each 30-minute webcast, Kevin will answer your questions live and/or teach additional concepts outside the course material.

Webcast attendance is optional. You are welcome to post your questions ahead of time, and watch the webcast recording (and read the chat transcript) afterwards.

Week | Sunday at 8:00pm ET | Tuesday at 1:00pm ET
--- | --- | ---
1 | [October 2](https://www.crowdcast.io/e/master-course/1) | [October 4](https://www.crowdcast.io/e/master-course/2)
2 | [October 9](https://www.crowdcast.io/e/master-course/3) | [October 11](https://www.crowdcast.io/e/master-course/4)
3 | October 16 | October 18
4 | October 23 | October 25
5 | October 30 | November 1
6 | November 6 | November 8
7 | November 13 | November 15
8 | November 20 | November 22

### Submitting Homeworks

If you would like to submit your homework for review, please post it online (ideally as a Jupyter notebook) and send a link to Alex Egorenkov via Slack. Watch [tips for submitting your homework](https://www.youtube.com/watch?v=9DxttSSkl60) (6 minutes) to learn how to post your homework online using [Gist](https://gist.github.com/) and how to make your Jupyter notebook readable.

-----

### Before the Course

* Make sure that [scikit-learn](http://scikit-learn.org/stable/install.html), [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), and [matplotlib](http://matplotlib.org/users/installing.html) (and their dependencies) are installed on your system. The easiest way to accomplish this is by downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Both Python 2 and 3 are welcome.
* If you are not familiar with Git and GitHub, watch my [quick introduction to Git and GitHub](https://www.youtube.com/watch?v=zYG8B8q722g) (8 minutes). Note that the repository shown in the video is from a previous iteration of the course, and the GitHub interface has also changed slightly.
    * For a longer introduction to Git and GitHub, watch my [11-video series](https://www.youtube.com/playlist?list=PL5-da3qGB5IBLMp7LtN8Nc3Efd4hJq0kD) (36 minutes).
* If you are not familiar with the Jupyter notebook, watch my [introductory video](https://www.youtube.com/watch?v=IsXXlYVBt1M&t=4m57s) (8 minute segment). Note that the Jupyter notebook was previously called the "IPython notebook", and the interface has also changed slightly. (Here is the [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/02_machine_learning_setup.ipynb) shown in the video.)
* If you are not yet comfortable with scikit-learn, review the notebooks and/or videos from my [scikit-learn video series](https://github.com/justmarkham/scikit-learn-videos), focusing specifically on the following topics:
    * Machine learning terminology, and working with data in scikit-learn ([video 3](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/03_getting_started_with_iris.ipynb))
    * scikit-learn's 4-step modeling pattern ([video 4](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/04_model_training.ipynb))
    * Train/test split ([video 5](https://www.youtube.com/watch?v=0pP4EwWJgIU&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=5), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/05_model_evaluation.ipynb))
    * Accuracy, confusion matrix, and AUC ([video 9](https://www.youtube.com/watch?v=85dtiMz9tSo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=9), [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb))
* If you are not yet comfortable with pandas, review the notebook and/or videos from my [pandas video series](https://github.com/justmarkham/pandas-videos). Alternatively, review another one of my [recommended pandas resources](http://www.dataschool.io/best-python-pandas-resources/).

-----

### Week 1: Working with Text Data in scikit-learn

**Topics covered:**
* Model building in scikit-learn (refresher)
* Representing text as numerical data
* Reading the SMS data
* Vectorizing the SMS data
* Building a Naive Bayes model
* Comparing Naive Bayes with logistic regression
* Calculating the "spamminess" of each token
* Creating a DataFrame from individual text files

**Video recordings:**
* [Part 1](https://www.youtube.com/watch?v=Qtbrdvc2ags&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=1) (1:08:05)
* [Part 2](https://www.youtube.com/watch?v=G6G7BMkQLY4&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=2) (52:33)

**Before class:**
* Read Paul Graham's classic post, [A Plan for Spam](http://www.paulgraham.com/spam.html), for an overview of a basic text classification system (using a Bayesian approach).
* Read this brief Quora post on [airport security](http://www.quora.com/In-laymans-terms-how-does-Naive-Bayes-work/answer/Konstantin-Tt) for an intuitive explanation of how Naive Bayes classification works.
* Watch [What is Text Classification?](https://www.youtube.com/watch?v=c3fnHA6yLeY&list=PL6397E4B26D00A269&index=24) (8 minutes) and [Naive Bayes](https://www.youtube.com/watch?v=DdYSMwEWbd4&list=PL6397E4B26D00A269&index=25) (3 minutes) from Coursera's NLP course. (Here are the [slides](http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf) used in the videos.)

**During class:**
* Working with Text Data in scikit-learn ([notebook](notebooks/01_text_data.ipynb), [notebook with output](notebooks/01_text_data_updated.ipynb), [script](scripts/01_text_data.py))
    * Documentation: [Text feature extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
    * Dataset: [SMS spam collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

**After class:**
* Homework with Yelp reviews data ([notebook](notebooks/01_yelp_homework.ipynb), [script](scripts/01_yelp_homework.py))
    * Dataset: Subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition
    * Solution ([notebook](notebooks/01_yelp_homework_solution.ipynb), [script](scripts/01_yelp_homework_solution.py))

**Resources:**
* The scikit-learn documentation describes the [performance trade-offs](http://scikit-learn.org/stable/modules/computational_performance.html#influence-of-the-input-data-representation) involved when choosing between sparse and dense input data representations.
* If you enjoyed Paul Graham's article, you can read [his follow-up article](http://www.paulgraham.com/better.html) on how he improved his spam filter and this [related paper](http://www.merl.com/publications/docs/TR2004-091.pdf) about state-of-the-art spam filtering in 2004.
* If you like cheat sheets, this is a well-organized 10-page [Cheat sheet to the pandas DataFrame object](https://drive.google.com/file/d/0ByIrJAE4KMTtTUtiVExiUGVkRkE/view).
* For an introduction to Naive Bayes, read Sebastian Raschka's article on [Naive Bayes and Text Classification](http://sebastianraschka.com/Articles/2014_naive_bayes_1.html). As well, Wikipedia has two excellent articles ([Naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Naive Bayes spam filtering](http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)), and Cross Validated has a good [Q&A](http://stats.stackexchange.com/questions/21822/understanding-naive-bayes).
* For an introduction to logistic regression, read my [lesson notebook](http://nbviewer.jupyter.org/github/justmarkham/DAT8/blob/master/notebooks/12_logistic_regression.ipynb).
* For a comparison of Naive Bayes and logistic regression (and other classifiers), read the [Supervised learning superstitions cheat sheet](http://ryancompton.net/assets/ml_cheat_sheet/supervised_learning.html) or browse my [comparison table](http://www.dataschool.io/comparing-supervised-learning-algorithms/). Also, this [paper by Andrew Ng](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) compares the performance of Naive Bayes and logistic regression across a variety of datasets, demonstrating that logistic regression tends to have a lower asymptotic error than Naive Bayes.
* The scikit-learn documentation on [probability calibration](http://scikit-learn.org/stable/modules/calibration.html) explains what it means for a predicted probability to be calibrated, and my blog post on [click-through rate prediction with logistic regression](https://web.archive.org/web/20160420174205/http://blog.dato.com/beginners-guide-to-click-through-rate-prediction-with-logistic-regression) explains why calibrated probabilities are useful in the real world.

-----

### Week 2: Basic Natural Language Processing (NLP)

**Topics covered:**
* What is NLP?
* Reading in the Yelp reviews corpus
* Tokenizing the text
* Comparing the accuracy of different approaches
* Removing frequent terms (stop words)
* Removing infrequent terms
* Handling Unicode errors

**Video recordings:**
* [Part 1](https://www.youtube.com/watch?v=QaWYpEVzjkk&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=3) (1:05:32)
* [Part 2](https://www.youtube.com/watch?v=TB8ptQ35Wzo&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=4) (1:05:39)

**Before class:**
* Watch [Introduction to NLP](https://www.youtube.com/watch?v=nfoudtpBV68&list=PL6397E4B26D00A269&index=1) (13 minutes) from Coursera's NLP course. (Here are the [slides](http://spark-public.s3.amazonaws.com/nlp/slides/intro.pdf) used in the video.)
* Read the [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), focusing on the Parameters section, for a preview of some of the options we will tune during class.

**During class:**
* Basic Natural Language Processing ([notebook](notebooks/02_basic_nlp.ipynb), [notebook with output](notebooks/02_basic_nlp_updated.ipynb), [script](scripts/02_basic_nlp.py))
    * Documentation: [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [Decoding text files](http://scikit-learn.org/stable/modules/feature_extraction.html#decoding-text-files)
    * Dataset: Subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition

**After class:**
* Homework with McDonald's sentiment data ([notebook](notebooks/02_mcdonalds_homework.ipynb), [script](scripts/02_mcdonalds_homework.py))
    * Dataset: McDonald's customer comments annotated by [CrowdFlower](http://www.crowdflower.com/data-for-everyone)

**Resources:**
* [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) is a classic paper that discusses multiple topics relevant to this week's class: overfitting, feature engineering, and the curse of dimensionality.
* This slide deck concisely defines many of the [key NLP terms](https://github.com/ga-students/DAT_SF_9/blob/master/16_Text_Mining/DAT9_lec16_Text_Mining.pdf).
* [Automatically Categorizing Yelp Businesses](http://engineeringblog.yelp.com/2015/09/automatically-categorizing-yelp-businesses.html) discusses how Yelp uses NLP and scikit-learn to solve the problem of uncategorized businesses.
* [Putting 1 million new words into the dictionary](https://www.youtube.com/watch?v=sum5Hq2FTsw) (42 minutes) presents a Python-based text classification pipeline for locating the definitions of previously undefined words.
* [How to Read the Mind of a Supreme Court Justice](http://fivethirtyeight.com/features/how-to-read-the-mind-of-a-supreme-court-justice/) discusses CourtCast, a machine learning model that predicts the outcome of Supreme Court cases using text-based features only. (The CourtCast creator wrote a post explaining [how it works](https://sciencecowboy.wordpress.com/2015/03/05/predicting-the-supreme-court-from-oral-arguments/), and the [Python code](https://github.com/nasrallah/CourtCast) is available on GitHub.)
* [Identifying Humorous Cartoon Captions](http://www.cs.huji.ac.il/~dshahaf/pHumor.pdf) is a readable paper about identifying funny captions submitted to the New Yorker Caption Contest.
* The Unicode Consortium has helpful materials on the Unicode standard, including a [Technical Introduction](http://www.unicode.org/standard/principles.html) and an incredibly detailed [tutorial](http://www.unicode.org/notes/tn23/Muller-Slides+Narr.pdf).
* [The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets](http://www.joelonsoftware.com/articles/Unicode.html) is a classic post by Joel Spolsky. [Unicode in Python, Completely Demystified](http://farmdev.com/talks/unicode/), Ned Batchelder's [Pragmatic Unicode](http://nedbatchelder.com/text/unipain.html), the [Unicode HOWTO](https://docs.python.org/2/howto/unicode.html) from Python's documentation, and this [StackOverflow Q&A](http://stackoverflow.com/questions/11339955/python-string-encode-decode) are also good resources.
* If you are a fan of emoji, you may enjoy browsing the Unicode Code Charts for [Emoticons](http://www.unicode.org/charts/PDF/U1F600.pdf) and [Miscellaneous Symbols and Pictographs](http://www.unicode.org/charts/PDF/U1F300.pdf), reading about [how emoji get added to the Unicode standard](http://blog.emojipedia.org/unicode-8-what-and-when/), or viewing the [emoji candidates for future Unicode releases](http://emojipedia.org/emojipedia/sample-images/).
* The scikit-learn documentation on [decoding text files](http://scikit-learn.org/stable/modules/feature_extraction.html#decoding-text-files) contains good advice for dealing with encoding errors.
