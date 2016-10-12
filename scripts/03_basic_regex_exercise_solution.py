# # Basic Regex Exercises (Solution)

# for Python 2: use print only as a function
from __future__ import print_function


# ## Exercise 1: Baltimore homicides
# 
# A Baltimore newspaper's website contains a map of all homicides in the city, including details about the victims. That map has been scraped, and the data is stored in the file **`homicides.txt`** (in the **`data`** directory of the course repository).

# read the data into a list (each row is one list element)
with open('../data/homicides.txt') as f:
    data = [row for row in f]


# check the number of rows
len(data)


# examine the first 5 rows
data[0:5]


# We want a list of the **ages** of the Baltimore homicide victims. (If the age is missing, insert a zero instead.) Here is the **expected output:**
# 
# > `ages = ['17 years old', '26 years old', ..., '0 years old', ...]`

import re


ages = []
for row in data:
    match = re.search(r'\d+ years? old', row)
    if match:
        ages.append(match.group())
    else:
        ages.append('0 years old')


print(ages[0:100])


# As a **bonus task**, use the `ages` list to create a second list of integers. (This does not require regular expressions.) Here is the **expected output:**
# 
# > `age_nums = [17, 26, ..., 0, ...]`

# split the string on spaces, only keep the first element, and convert to int
age_nums = [int(element.split()[0]) for element in ages]
print(age_nums[0:100])


# check that all of the objects are the same length
assert(len(data) == len(ages) == len(age_nums))


# As **another bonus task**, use the `age_nums` list to calculate the **mean age** of a homicide victim (excluding zeros).

# remove zero ages
clean_age_nums = [num for num in age_nums if num != 0]
len(clean_age_nums)


# calculate the mean age (excluding zeros)
sum(clean_age_nums) / float(len(clean_age_nums))


# ## Exercise 2: Baltimore homicides, revisited
# 
# Using **match groups**, create the `age_nums` list directly from the regular expression. Here is the **expected output:**
# 
# > `age_nums = [17, 26, ..., 0, ...]`

age_nums = []
for row in data:
    match = re.search(r'(\d+) years? old', row)
    if match:
        age_nums.append(int(match.group(1)))
    else:
        age_nums.append(0)


print(age_nums[0:100])

