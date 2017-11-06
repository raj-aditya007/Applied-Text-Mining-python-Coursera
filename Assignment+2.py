
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[ ]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[ ]:

def answer_one():
    
    
    return len(set(nltk.word_tokenize(moby_raw)))/len(nltk.word_tokenize(moby_raw))

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[ ]:

def answer_two():
    from nltk.book import FreqDist
    token_dict = FreqDist(moby_tokens)
    return (((token_dict['whale'] + token_dict['Whale'])*100)/float(len(nltk.word_tokenize(moby_raw))))

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[9]:

def answer_three():
    
    from nltk.book import FreqDist
    import operator
    token_dict = FreqDist(moby_tokens)
    sorted_token_dict = sorted(token_dict.items(), key=operator.itemgetter(1))
    lst = sorted_token_dict[-20:]
    lst.reverse()
    return lst
answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[11]:

def answer_four():
    
    from nltk.book import FreqDist
    token_dict = FreqDist(moby_tokens)
    res_lis = []
    for w in token_dict.keys() :
        if len(w) > 5 and token_dict[w] > 150 :
            res_lis.append(w)
    res_lis.sort()
    return res_lis
answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[ ]:

def answer_five():
    
    from nltk.book import FreqDist
    token_dict = FreqDist(moby_tokens)
    max_len = 0
    for w in token_dict.keys() :
        if len(w) > max_len  :
            max_word = w
            max_len = len(w)
    res_tuple = (max_word ,max_len)
    return res_tuple
answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[3]:

def answer_six():
    import operator
    from nltk.book import FreqDist
    token_dict  =FreqDist(moby_tokens)
    res_lis = {}
    for w in token_dict.keys() :
        if w.isalpha() and token_dict[w] > 2000 :
            res_lis[w] = token_dict[w]    
    sorted_res_list = sorted(res_lis.items(), key=operator.itemgetter(1))
    sorted_res_list.reverse()
    result = [(f,w) for w,f in sorted_res_list]
    return result
answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[ ]:

def answer_seven():
    
    sen_tokens = nltk.sent_tokenize(moby_raw)
    return len(moby_tokens)/len(sen_tokens)

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[ ]:

def answer_eight():
    import collections
    pos_token = nltk.pos_tag(text1)
    pos_counts = collections.Counter((subl[1] for subl in pos_token))
    return pos_counts.most_common(5)

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[ ]:

from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.metrics.distance import (
    jaccard_distance,
    )
    from nltk.util import ngrams
    spellings_series = pd.Series(correct_spellings)
    correct = []
    for entry in entries :
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = ((jaccard_distance(set(ngrams(entry, 3)),set(ngrams(word, 3))), word) for word in spellings)
        closet = min(distances)
        correct.append(closet[1])
        
    return correct
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.metrics.distance import (
    jaccard_distance,
    )
    from nltk.util import ngrams
    spellings_series = pd.Series(correct_spellings)
    correct = []
    for entry in entries :
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = ((jaccard_distance(set(ngrams(entry, 4)),set(ngrams(word, 4))), word) for word in spellings)
        closet = min(distances)
        correct.append(closet[1])
        
    return correct
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[6]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.metrics.distance import (
    edit_distance,
    )
    spellings_series = pd.Series(correct_spellings)
    correct = []
    for entry in entries :
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = ((edit_distance(entry,word), word) for word in spellings)
        closet = min(distances)
        correct.append(closet[1])
        
    return correct
    
answer_eleven()

