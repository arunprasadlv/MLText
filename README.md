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
* [Week 3: Intermediate NLP and Basic Regular Expressions](#week-3-intermediate-nlp-and-basic-regular-expressions)
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
3 | [October 16](https://www.crowdcast.io/e/master-course/5) | [October 18](https://www.crowdcast.io/e/master-course/6)
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
* [Explanation of Week 2 homework solution](https://www.youtube.com/watch?v=T4zbsw6cl1s&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=5) (1:07:01)

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
    * Solution ([notebook](notebooks/02_mcdonalds_homework_solution.ipynb), [script](scripts/02_mcdonalds_homework_solution.py), [video explanation](https://www.youtube.com/watch?v=T4zbsw6cl1s&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=5))

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

-----

### Week 3: Intermediate NLP and Basic Regular Expressions

**Topics covered:**
* Intermediate NLP:
    * Reading in the Yelp reviews corpus
    * Term Frequency-Inverse Document Frequency (TF-IDF)
    * Using TF-IDF to summarize a Yelp review
    * Sentiment analysis using TextBlob
* Basic Regular Expressions:
    * Why learn regular expressions?
    * Rules for searching
    * Metacharacters
    * Quantifiers
    * Using regular expressions in Python
    * Match groups
    * Character classes
    * Finding multiple matches

**Video recordings:**
* [Part 1](https://www.youtube.com/watch?v=jEsEJlrn-io&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=6) on Intermediate NLP (56:51)
* [Part 2](https://www.youtube.com/watch?v=Vyn99hLGZwM&list=PL5-da3qGB5IBK1krD5d2P3MMCsnFQHrxj&index=7) on Basic Regular Expressions (1:27:21)

**Before class:**
* Install TextBlob by following [these instructions](https://textblob.readthedocs.org/en/dev/install.html), and then test that installation was successful by running `import textblob` from within Python.

**During class:**
* Intermediate Natural Language Processing ([notebook](notebooks/03_intermediate_nlp.ipynb), [notebook with output](notebooks/03_intermediate_nlp_updated.ipynb), [script](scripts/03_intermediate_nlp.py))
    * Documentation: [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [TextBlob](https://textblob.readthedocs.org/)
    * Dataset: Subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition
* Basic Regular Expressions
    * [Regular expressions 101](https://regex101.com/#python): real-time testing of regular expressions
    * Reference guide ([notebook](notebooks/03_basic_regex.ipynb), [notebook with output](notebooks/03_basic_regex_updated.ipynb), [script](scripts/03_basic_regex.py))
    * Class exercises ([notebook](notebooks/03_basic_regex_exercise.ipynb), [script](scripts/03_basic_regex_exercise.py))
    * Solution to class exercises ([notebook](notebooks/03_basic_regex_exercise_solution.ipynb), [script](scripts/03_basic_regex_exercise_solution.py))
    * Datasets: Baltimore homicides, email messages

**After class:**
* Using TF-IDF, write a document summarizer for a corpus of your choosing, but summarize using full sentences or paragraphs rather than individual words. (This approach is similar to the [SMMRY](http://smmry.com/about) algorithm, which is used by Reddit's [autotldr](https://np.reddit.com/r/autotldr/comments/31b9fm/faq_autotldr_bot/).)
    * Before beginning, you may want to watch the [four lectures on summarization](https://www.youtube.com/watch?v=EZLCOrrl0Wc&list=PL6397E4B26D00A269&index=98) (28 minutes) from Coursera's NLP course. (Here are the [slides](http://spark-public.s3.amazonaws.com/nlp/slides/summarization.pdf) used in the videos.)
    * You can use my lesson code as a starting point, or build something from scratch!
* Homework with basic regular expressions ([notebook](notebooks/03_basic_regex_homework.ipynb), [script](scripts/03_basic_regex_homework.py))
    * Datasets: [FAA tower closures](http://www.faa.gov/news/media/fct_closed.pdf), my Stack Overflow [reputation history](http://stackoverflow.com/reputation)

**NLP Resources:**
* Coursera's NLP course has [video lectures](https://www.youtube.com/playlist?list=PL6397E4B26D00A269) on sentiment analysis, TFIDF, summarization, and many other fundamental NLP topics. (Here are the [slides](http://web.stanford.edu/~jurafsky/NLPCourseraSlides.html) used in all of the videos.)
* [TF-IDF is about what matters](http://planspace.org/20150524-tfidf_is_about_what_matters/) explains how scikit-learn computes TF-IDF scores.
* FiveThirtyEight used TF-IDF to compute the most repeated talking points by the 2016 [Republican](http://fivethirtyeight.com/features/these-are-the-phrases-each-gop-candidate-repeats-most/) and [Democratic](http://fivethirtyeight.com/features/these-are-the-phrases-that-sanders-and-clinton-repeat-most/) presidential candidates. (Here is the [data and Python script](https://github.com/fivethirtyeight/data/tree/master/repeated-phrases-gop) for the Republican analysis.)
* [The Simpsons by the Data](http://toddwschneider.com/posts/the-simpsons-by-the-data/) uses TF-IDF on episode scripts from The Simpsons to create episode summaries. (Here is the [dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).)
* [Modern Methods for Sentiment Analysis](http://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis) shows how "word vectors" can be used for more accurate sentiment analysis.
* In [Sentiment Classification Using scikit-learn](https://www.youtube.com/watch?v=y3ZTKFZ-1QQ) (50 minutes), Facebook explains how they detect sentiment by training a Naive Bayes model on emoji-labeled data.
* [Natural Language Processing with Python](http://www.nltk.org/book/) is the most popular book for going in-depth with the [Natural Language Toolkit](http://www.nltk.org/) (NLTK).
* [spaCy](https://spacy.io/) is a newer Python library for text processing that is focused on performance (unlike NLTK).
* If you want to get serious about NLP, [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) is a suite of tools (written in Java) that is highly regarded.

**Regular Expressions Resources:**
* Google's Python Class includes an excellent [introductory lesson](https://developers.google.com/edu/python/regular-expressions) on regular expressions (which also has an associated [video](https://www.youtube.com/watch?v=kWyoYtvJpe4&index=4&list=PL5-da3qGB5IA5NwDxcEJ5dvt8F9OQP7q5)).
* Python for Informatics has a nice [chapter](http://www.pythonlearn.com/html-270/book012.html) on regular expressions. (If you want to run the examples, you'll need to download [mbox.txt](http://www.py4inf.com/code/mbox.txt) and [mbox-short.txt](http://www.py4inf.com/code/mbox-short.txt).)
* [5 Tools You Didn't Know That Use Regular Expressions](http://blog.codeschool.io/2015/07/30/5-tools-you-didnt-know-that-use-regular-expressions/) demonstrates how regular expressions can be used with Excel, Word, Google Spreadsheets, Google Forms, text editors, and other tools.
* This relatively simple [regular expression pattern](http://sentiment.christopherpotts.net/tokenizing.html#emoticons) can apparently capture 96% of emoticons used on Twitter.
