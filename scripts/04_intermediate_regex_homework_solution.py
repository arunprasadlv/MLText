# # Intermediate Regex Homework (Solution)

# for Python 2: use print only as a function
from __future__ import print_function


# ## UFO sightings
# 
# The [ufo-reports](https://github.com/planetsig/ufo-reports) GitHub repository contains reports of UFO sightings downloaded from the [National UFO Reporting Center](http://www.nuforc.org/) website. One of the data fields is the **duration of the sighting**, which includes **free-form text**. These are some example entries:
# 
# - 45 minutes
# - 1-2 hrs
# - 20 seconds
# - 1/2 hour
# - about 3 mins
# - several minutes
# - one hour?
# - 5min

# Here is **how to read in the file:**
# 
# - Use the pandas **`read_csv()`** function to read directly from this [URL](https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv).
# - Use the **`header=None`** parameter to specify that the data does not have a header row.
# - Use the **`nrows=100`** parameter to specify that you only want to read in the first 100 rows.
# - Save the relevant Series as a Python list, just like we did in a class exercise.

# Your assignment is to **normalize the duration data for the first 100 rows** by splitting each entry into two parts:
# 
# - The first part should be a **number**: either a whole number (such as '45') or a decimal (such as '0.5').
# - The second part should be a **unit of time**: either 'hr' or 'min' or 'sec'
# 
# The expected output is a **list of tuples**, containing the **original (unedited) string**, the **number**, and the **unit of time**. Here is a what the output should look like:
# 
# > `clean_durations = [('45 minutes', '45', 'min'), ('1-2 hrs', '1', 'hr'), ('20 seconds', '20', 'sec'), ...]`

# Here are the **"rules" and guiding principles** for this assignment:
# 
# - The normalized duration does not have to be exactly correct, but it must be at least **within the given range**. For example:
#     - If the duration is '20-30 min', acceptable answers include '20 min' and '30 min'.
#     - If the duration is '1/2 hour', the only acceptable answer is '0.5 hr'.
# - When a number is not given, you should make a **"reasonable" substitution for the words**. For example:
#     - If the duration is 'several minutes', you can approximate this as '5 min'.
#     - If the duration is 'couple minutes', you can approximate this as '2 min'.
# - You are not allowed to **skip any entries**. (Your list of tuples should have a length of 100.)
# - Try to use **as few substitutions as possible**, and make your regular expression **as simple as possible**.
# - Just because you don't get an error doesn't mean that your code was successful. Instead, you should **check each entry by hand** to see if it produced an acceptable result.

# **Bonus tasks:**
# 
# - Try reading in **more than 100 rows**, and see if your code still produces the correct results.
# - When a range is specified (such as '1-2 hrs' or '10 to 15 sec'), **calculate the exact midpoint** ('1.5 hr' or '12.5 sec') to use in your normalized data.

import pandas as pd
url = 'https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv'
ufo = pd.read_csv(url, header=None, nrows=100)
ufo.head()


# save the '6' Series as a Python list
durations = ufo[6].tolist()
print(durations)


import re


# find a whole number or a decimal (such as '20' or '4.5')
pattern_number = re.compile(r'[\d.]+')

# find a fraction (such as '1/2')
pattern_fraction = re.compile(r'(\d+)/(\d+)')

# find a range (such as '20-30' or '20 to 30')
pattern_range = re.compile(r'(\d+)(-| to )(\d+)')

# find a word that needs a substitution
pattern_substitution = re.compile(r'several|couple|few|one')
substitutions = {'several':'5', 'couple':'2', 'few':'2', 'one':'1'}

# find a time unit
pattern_time = re.compile(r'hour|hr|min|sec')


clean_durations = []

for text in durations:
    
    # search for each pattern and store the resulting match object (will be 'None' if not found)
    match_number = pattern_number.search(text)
    match_fraction = pattern_fraction.search(text)
    match_range = pattern_range.search(text)
    match_substitution = pattern_substitution.search(text)
    match_time = pattern_time.search(text)
    
    if match_range:
        # calculate the midpoint of the range
        range_start = float(match_range.group(1))
        range_end = float(match_range.group(3))
        number = str((range_start + range_end) / 2)
    elif match_fraction:
        # convert the fraction into a decimal
        numerator = float(match_fraction.group(1))
        denominator = float(match_fraction.group(2))
        number = str(numerator / denominator)
    elif match_number:
        # store the first number found
        number = match_number.group()
    elif match_substitution:
        # convert the word into a number
        number = substitutions[match_substitution.group()]
    else:
        # flag any instances for which no number was found
        number = 'not found'
    
    if match_time:
        # find the time unit and standardize it
        time = match_time.group()
        standard_time = re.sub(r'hour', r'hr', time)
    else:
        # flag any instances for which no time unit was found
        standard_time = 'not found'
    
    # save the results
    clean_durations.append((text, number, standard_time))

clean_durations


# **Remarks:** My approach does very well with the first 100 rows, though it does not look for multiple time units in the same text (such as '1min. 39s').

# examine the next 100 durations
ufo_next = pd.read_csv(url, header=None, skiprows=100, nrows=100)
print(ufo_next[6].tolist())


# **Remarks:** My approach was tuned to the first 100 rows, and thus it would need to be further adjusted to handle the following cases:
# 
# - **less then a minute**: I need to substitute '1' for 'a'.
# - **00:43** and **5:00** and **1:00:00**: I need to handle cases in which the time unit is not stated explicitly.
# - **1 1/2 hr.**: I need to handle mixed numbers.
# - **2 - 3 minutes**: I need to allow for spaces around the dash in time ranges.
# - **5/6 minutes**: I need to handle cases in which apparent fractions are actually intended to represent time ranges.
# - **1 to 1 1/2 minutes**: I need to handle ranges that include mixed numbers.
# 
# I would guess that if I continued to tune my approach to appropriately handle nearly all of the first 1000 rows, it would produce highly accurate (though not perfect) results for the remaining rows.
