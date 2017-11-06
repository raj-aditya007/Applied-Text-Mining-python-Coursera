
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[2]:

import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)


# In[3]:

def date_sorter():
    import re
    from calendar import month_name
    import dateutil.parser
    from datetime import datetime

    df = pd.DataFrame(doc, columns=['text'])
    pattern = "[,.]? \d{4}|".join(month_name[1:]) + "[,.]? \d{4}";

    df['text'] = df['text'].apply(lambda x: x.strip('\n'))

    df['date'] = df['text'].apply(lambda x:re.findall(r'\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}\-\d{1,2}\-\d{2,4}|[A-Z][a-z]+\-\d{1,2}\-\d{4}|[A-Z][a-z]+[,.]? \d{2}[a-z]*,? \d{4}|\d{1,2} [A-Z][a-z,.]+ \d{4}|[A-Z][a-z]{2}[,.]? \d{4}|'+pattern+r'|\d{1,2}\/\d{4}|\d{4}',x))
    df['date'][271] = [df['date'][271][1]]
    df['date'] = df['date'].apply(lambda x: x[0])
    df['date'][461] = re.findall(r'\d{4}',df['date'][461])[0]
    df['date'][465] = re.findall(r'\d{4}',df['date'][465])[0]

    date_list = list(df['date'])

    i=0
    for year in date_list:
        if(re.match(r'\d{4}',year)) :
            #print(year)
            date_list[i] = 'January 1, '+date_list[i]
            year = date_list[i]
        elif (re.match(r'\d{1,2}\/\d{4}',year)) :
            date_split = year.split('/')
            date_list[i] = date_split[0] + '/1/'+date_split[1]
            year = date_list[i]
        elif(re.match(r'[A-Z][a-z]+[,.]? \d{4}',year)) :
            date_split = year.split(' ')
            date_list[i] = date_split[0] + ' 1 '+date_split[1]
            year = date_list[i]
        date_list[i] = dateutil.parser.parse(date_list[i]).strftime("%m/%d/%Y")
        i = i+1

    df['date'] = date_list
    fun = lambda date: datetime.strptime(date, "%m/%d/%Y")
    df['index'] = sorted(range(len(date_list)), key=lambda x : fun(date_list[x]))
    sdf = df
    sdf.drop('text', axis=1,inplace=True)
    #result = sdf.sort_values(by='date',axis=0, ascending=False, kind='mergesort')
    final = list(sdf['index'])
    final_series = pd.Series(final)    
    return final_series

