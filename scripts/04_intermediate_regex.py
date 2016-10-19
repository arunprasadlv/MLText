# # Intermediate Regular Expressions (Regex)

# ## Agenda
# 
# 1. Greedy or lazy quantifiers
# 2. Alternatives
# 3. Substitution
# 4. Anchors
# 5. Option flags
# 6. Lookarounds
# 7. Assorted functionality

# for Python 2: use print only as a function
from __future__ import print_function


import re


# ## Part 1: Greedy or lazy quantifiers
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


# Add a **`?`** after **`+`** or **`*`** to make them **"lazy"**, meaning that they try to use up as little of the string as possible:

re.search(r'<.+?>', s).group()


# ## Part 2: Alternatives
# 
# Alternatives define **multiple possible patterns** that can be used to produce a match. They are separated by a pipe and put in parentheses:

s = 'I live at 100 First St, which is around the corner.'


re.search(r'\d+ .+ (Ave|St|Rd)', s).group()


# ## Part 3: Substitution
# 
# `re.sub()` finds **all matches** in a given string and **replaces them** with a specified string:

s = 'my twitter is @jimmy, my emails are john@hotmail.com and jim@yahoo.com'


re.sub(r'jim', r'JIM', s)


re.sub(r' @\w+', r' @johnny', s)


# The replacement string can refer to text from **match groups**:
# 
# - `\1` refers to `group(1)`
# - `\2` refers to `group(2)`
# - etc.

re.sub(r'(\w+)@[\w.]+', r'\1@gmail.com', s)


# ## Part 4: Anchors
# 
# Anchors define **where in a string** the regular expression pattern must occur.
# 
# Anchor|What it requires
# ---|---
# **`^abc`** | this pattern must appear at the start of a string
# **`abc$`** | this pattern must appear at the end of a string

s = 'sid is missing class'


re.search(r'\w+', s).group()


re.search(r'\w+$', s).group()


# this will cause an error
# re.search(r'^is', s).group()


# ## Part 5: Option flags
# 
# Options flags **change the default behavior** of the pattern matching.
# 
# Default behavior | Option flag | Behavior when using flag
# ---|---|---
# matching is case-sensitive | re.IGNORECASE | matching is case-insensitive
# **`.`** matches any character except a newline | re.DOTALL | **`.`** matches any character including a newline
# within a multi-line string, **`^`** and **`$`**<br>match start and end of entire string | re.MULTILINE | **`^`** and **`$`** matches start and end of each line
# spaces and **`#`** are treated as literal characters | re.VERBOSE | spaces and **`#`** are ignored (except in a character class or<br>when preceded by **`\`**), and characters after **`#`** are ignored

s = 'LINE one\nLINE two'


print(s)


# case-sensitive
re.search(r'..n.', s).group()


# case-insensitive
re.search(r'..n.', s, flags=re.IGNORECASE).group()


# . does not match a newline
re.search(r'n.+', s).group()


# . matches a newline
re.search(r'n.+', s, flags=re.DOTALL).group()


# combine option flags
re.search(r'n.+', s, flags=re.IGNORECASE|re.DOTALL).group()


# $ matches end of entire string
re.search(r'..o\w*$', s).group()


# $ matches end of each line
re.search(r'..o\w*$', s, flags=re.MULTILINE).group()


# spaces are literal characters
re.search(r' \w+', s).group()


# spaces are ignored
re.search(r' \w+', s, flags=re.VERBOSE).group()


# use multi-line patterns and add comments in verbose mode
re.search(r'''
\     # single space
\w+   # one or more word characters
''', s, flags=re.VERBOSE).group()


# ## Part 6: Lookarounds
# 
# A **lookahead** matches a pattern only if it is **followed by** another pattern. For example:
# 
# - `100(?= dollars)` matches `'100'` only if it is followed by `' dollars'`
# 
# A **lookbehind** matches a pattern only if it is **preceded by** another pattern. For example:
# 
# - `(?<=\$)100` matches `'100'` only if it is preceded by `'$'`

s = 'Name: Cindy, 66 inches tall, 30 years old'


# find the age, without a lookahead
re.search(r'(\d+) years? old', s).group(1)


# find the age, with a lookahead
re.search(r'\d+(?= years? old)', s).group()


# find the name, without a lookbehind
re.search(r'Name: (\w+)', s).group(1)


# find the name, with a lookbehind
re.search(r'(?<=Name: )\w+', s).group()


# ## Part 7: Assorted functionality

# `re.compile()` compiles a regular expression pattern for **improved readability and performance** (if the pattern is used frequently):

s = 'emails: john-doe@gmail.com and jane-doe@hotmail.com'


email = re.compile(r'[\w.-]+@[\w.-]+')


# these are all equivalent
re.search(r'[\w.-]+@[\w.-]+', s).group()
re.search(email, s).group()
email.search(s).group()


# these are all equivalent
re.findall(r'[\w.-]+@[\w.-]+', s)
re.findall(email, s)
email.findall(s)


# Use the `span()` method of a match object, rather than the `group()` method, to determine the **location of a match**:

re.search(email, s).span()


s[8:26]


# `re.split()` **splits a string** by the occurrences of a regular expression pattern:

re.split(r'john|jane', s)

