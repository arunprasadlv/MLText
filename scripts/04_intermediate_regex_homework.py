# # Intermediate Regex Homework

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
