# # Basic Regular Expressions (Regex)

# ## Agenda
# 
# 1. Rules for searching
# 2. Using regex in Python
# 3. Metacharacters
# 4. Quantifiers
# 5. Match groups
# 6. Character classes
# 7. Finding multiple matches

# for Python 2: use print only as a function
from __future__ import print_function


# ## Part 1: Rules for searching
# 
# - **Ordinary characters** (also known as literals) match themselves exactly
# - **Case-sensitive** (by default)
# - Search proceeds through the string from start to end, **stopping at first match**
# - The **entire pattern** must match a continuous sequence of characters

# ## Part 2: Using regex in Python

# use built-in regex module
import re


# define the string to search
s = 'my 1st string!!'


# pattern as 'raw string', then string to search, returns 'match object'
re.search(r'st', s)


# access the results using the 'group' method
re.search(r'st', s).group()


# returns 'None' if no match is found
re.search(r'sti', s)


# causes an error since 'None' does not have a 'group' method
# re.search(r'sti', s).group()


# better error handling
match = re.search(r'st', s)
if match:
    print(match.group())


# does not cause an error since condition fails
match = re.search(r'sti', s)
if match:
    print(match.group())


# ## Part 3: Metacharacters
# 
# Metacharacters are the **opposite of literal characters**, because they represent something other than themselves.
# 
# Metacharacter | What it matches
# --- | ---
# **`.`** | any character except newline \n
# **`\w`** | word character (letter, digit, underscore)
# **`\W`** | non-word character
# **`\d`** | digit (0 through 9)
# **`\s`** | whitespace character (space, newline, return, tab, form)
# **`\S`** | non-whitespace character
# **`\.`** | period (you must escape a special character to match it)

s = 'my 1st string!!'


re.search(r'..', s).group()


re.search(r'..t', s).group()


re.search(r'\w\w', s).group()


re.search(r'\w\w\w', s).group()


re.search(r'\W', s).group()


re.search(r'\W\W', s).group()


re.search(r'\W\wt', s).group()


re.search(r'\d..', s).group()


# ## Part 4: Quantifiers
# 
# Quantifiers modify the **required quantity** of a character or a pattern.
# 
# Quantifier|What it matches
# ---|---
# **`a+`** | 1 or more occurrences of 'a' (the pattern directly to its left)
# **`a*`** | 0 or more occurrences of 'a'
# **`a?`** | 0 or 1 occurrence of 'a'

s = 'sid is missing class'


re.search(r'miss\w+', s).group()


re.search(r'is\w*', s).group()


re.search(r'is\w+', s).group()


# **`+`** and **`*`** are **"greedy"**, meaning that they try to use up as much of the string as possible:

s = 'Some text <h1>my heading</h1> More text'


re.search(r'<.+>', s).group()


# Quantifier|What it matches
# ---|---
# **`a{3}`** | exactly 3 occurrences of 'a'
# **`a{3,}`** | 3 or more occurrences of 'a'
# **`a{1,3}`** | 1 to 3 occurrences of 'a'

s = 'Sales on November 14: $250 for item 54321'


re.search(r'\d{3}', s).group()


re.search(r'\d{4,5}', s).group()


# ## Part 5: Match groups
# 
# Parentheses create **logical groups** inside of match text:
# 
# - `match.group()` corresponds to **entire match text** (as usual)
# - `match.group(1)` corresponds to **first group**
# - `match.group(2)` corresponds to **second group**
# 
# **Note:** There is no limit to the number of groups you can create.

s = 'my 1st string!!'


re.search(r'\d..', s).group()


re.search(r'(\d)(..)', s).group()


re.search(r'(\d)(..)', s).group(1)


re.search(r'(\d)(..)', s).group(2)


# When using match groups, you don't have to include every part of the pattern inside a group:

re.search(r'\d(..)', s).group(1)


# ## Part 6: Character classes
# 
# Brackets indicate a **set of characters** (known as a "character class"), any one of which can be used to make a regular expression match.
# 
# Character class|What it matches
# ---|---
# **`[abc]`** | a or b or c
# **`[abc-]`** | a or b or c or dash
# **`[a-z]`** | any lowercase letter (dash indicates a range, unless it's the last character)
# **`[\w.]`** | any word character or period (**`.`** just means a literal period)
# **`[^abc]`** | anything except a or b or c

s = 'my email is john-doe@gmail.com, okay?'


re.search(r'\w+@\w+', s).group()


re.search(r'[\w.-]+@[\w.-]+', s).group()


# ## Part 7: Finding multiple matches
# 
# `re.findall()` finds **all matches** in a given string and returns them as a **list of strings**.

s = 'emails: john-doe@gmail.com, jane-doe@hotmail.com'


re.findall(r'[\w.-]+@[\w.-]+', s)


# If the regular expression pattern includes a **single match group**, only the matching portion is returned:

re.findall(r'([\w.-]+)@[\w.-]+', s)


# If the regular expression pattern includes **multiple match groups**, a list of tuples is returned:

re.findall(r'([\w.-]+)@([\w.-]+)', s)

