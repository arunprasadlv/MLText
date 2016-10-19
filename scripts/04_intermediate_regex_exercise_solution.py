# # Intermediate Regex Exercises (Solution)

# for Python 2: use print only as a function
from __future__ import print_function


# ## Exercise 1: IMDb top 100 movies
# 
# Data about the 100 highest rated movies has been been scraped from the IMDb website and stored in the file **`imdb_100.csv`** (in the **`data`** directory of the course repository).

# read the file into a DataFrame
import pandas as pd
path = '../data/imdb_100.csv'
imdb = pd.read_csv(path)


imdb.columns


# save the 'title' Series as a Python list
titles = imdb.title.tolist()


print(titles)


# Here are a few of the titles from this list:
# 
# > `titles = [..., "It's a Wonderful Life", 'The Usual Suspects', 'Se7en', ...]`
# 
# We want a revised list with the **initial article (A/An/The) removed**, without affecting the rest of the title. Here is the **expected output:**
# 
# > `clean_titles = [..., "It's a Wonderful Life", 'Usual Suspects', 'Se7en', ...]`

import re


# remove the initial article
clean_titles = [re.sub(r'^(A|An|The) ', r'', title) for title in titles]
print(clean_titles)


# As a **bonus task**, add the removed article to the end of the title. Here is the **expected output:**
# 
# > `better_titles = [..., "It's a Wonderful Life", 'Usual Suspects, The', 'Se7en', ...]`

# move the initial article to the end
better_titles = [re.sub(r'^(A|An|The) (.+)', r'\2, \1', title) for title in titles]
print(better_titles)


# ## Exercise 2: FAA tower closures, revisited
# 
# A list of FAA tower closures has been copied from a [PDF](http://www.faa.gov/news/media/fct_closed.pdf) into the file **`faa.txt`**, which is stored in the **`data`** directory of the course repository.

# read the file into a single string
with open('../data/faa.txt') as f:
    data = f.read()


# examine the first 300 characters
print(data[0:300])


# create a list of tuples containing the tower IDs and their states
print(re.findall(r'([A-Z]{3}) .+ ([A-Z]{2})', data))


# Without changing the output, make this regular expression pattern more readable by using the **`re.VERBOSE`** option flag and adding comments.

print(re.findall(r'''
([A-Z]{3})\    # match group 1 is ID, then space
.+\            # any characters, then space
([A-Z]{2})     # match group 2 is state
''', data, flags=re.VERBOSE))

