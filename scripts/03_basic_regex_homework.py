# # Basic Regex Homework

# for Python 2: use print only as a function
from __future__ import print_function


# ## Homework 1: FAA tower closures
# 
# A list of FAA tower closures has been copied from a [PDF](http://www.faa.gov/news/media/fct_closed.pdf) into the file **`faa.txt`**, which is stored in the **`data`** directory of the course repository.

# read the file into a single string
with open('../data/faa.txt') as f:
    data = f.read()


# check the number of characters
len(data)


# examine the first 500 characters
print(data[0:500])


# examine the last 500 characters
print(data[-500:])


# Your assignment is to **create a list of tuples** containing the **tower IDs** and the **states** they are located in.
# 
# Here is the **expected output:**
# 
# > `faa = [('DHN', 'AL'), ('TCL', 'AL'), ..., ('PKB', 'WV')]`

import re


# As a **bonus task**, use regular expressions to extract the **number of closures** listed in the second line of the file (149), and then use an **assertion** to check that the number of closures is equal to the length of the `faa` list.

# ## Homework 2: Stack Overflow reputation
# 
# I have downloaded my **Stack Overflow reputation history** into the file **`reputation.txt`**, which is stored in the **`data`** directory of the course repository. (If you are a Stack Overflow user with a reputation of 10 or more, you should be able to [download your own reputation history](http://stackoverflow.com/reputation).)
# 
# We are only interested in the lines that **begin with two dashes**, such as:
# 
# > `-- 2012-08-30 rep +5    = 6`
# 
# That line can be interpreted as follows: "On 2012-08-30, my reputation increased by 5, bringing my reputation total to 6."

# Your assignment is to **create a list of tuples** containing only these dated entries, including the **date**, **reputation change** (regardless of whether it is positive/negative/zero), and **running total**.
# 
# Here is the **expected output:**
# 
# > `rep = [('2012-08-30', '+5', '6'), ('2012-12-11', '+10', '16'), ...,  ('2015-10-14', '-1', '317')]`

# As a **bonus task**, convert this list of tuples into a **pandas DataFrame**. It should have appropriate column names, and the second and third columns should be of type integer (rather than string/object).
