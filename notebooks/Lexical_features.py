#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re 
import heapq 
import string

from tqdm import tqdm_notebook
from nltk import ngrams
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


# In[ ]:


## Get array of all lexical features
# If you want numeric flooding or punctuation features, make sure to set these parameters to True when calling the method
# bow_length is the length of your bag_of_words features.
def get_lexical_features(corpus, flooding_numeric=False, punctuation_numeric=False, bow_length=None):
    features = []
    
    # Token unigrams
    token_unigrams = corpus
    
    # Token bigrams
    token_bigrams = []
    
    # Character trigrams (with spaces)
    char_trigrams = []
    
    # Character fourgrams (with spaces)
    char_fourgrams = []
#     char_trigrams_nosp = []
#     char_fourgrams_nosp = []        
    
    # Punctuation (numerical/binary)
    punctuation = []
    
    # Capitalisation (numerical/binary)
    capitalisation = []
    
    # Flooding (numerical/binary)
    flooding = []
    
    # Hashtag frequency
    hashtag_freq = []
    
    # Hashtag-to-word ratio
    hashtag_to_word = []
    
    # Emoticon frequency
    emoticon_freq = []
    
    # Tweet length (in tokens)
    tweet_length = []
    
    # Count co-occurences in lists
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    # For every text in the corpus
    for tweet in corpus:
        
        # Save the length in tokens
        tweet_length.append(len(tweet))
        
        # Create sentences, with and without spaces
        sentence = " ".join(tweet)
        sentence_nospace = "".join(tweet)
        
        # Token bigrams
        token_bigrams.append(list(ngrams(tweet, 2)))
        
        # Character trigrams
        char_trigrams.append(list(ngrams(sentence, 3)))
        
        # Character fourgrams
        char_fourgrams.append(list(ngrams(sentence, 4)))
#         char_trigrams_nosp.append(list(ngrams(sentence_nospace, 3)))
#         char_fourgrams_nosp.append(list(ngrams(sentence_nospace, 4)))

        # Count punctuation and capitalisation
        amount_punct = count(sentence, string.punctuation)
        amount_cap = len(re.findall(r'[A-Z]',sentence))

        # If numeric, save amount of punctuation, else binary
        if amount_punct > 0:
            if punctuation_numeric:
                punctuation.append(amount_punct)
            else:
                punctuation.append(1)
        else:
            punctuation.append(0)

        # If numeric, save amount of capitalisation, else binary
        if amount_cap > 0:
            if punctuation_numeric:
                capitalisation.append(amount_cap)
            else:
                capitalisation.append(1)
        else:
            capitalisation.append(0)
        
        
        # Counters for flooding, hashtags and emoticons
        amount_flooding = 0
        amount_hashtags = 0
        amount_emoticons = 0
        
        # For every token
        for word in tweet:
            # Detect hashtags
            if word.startswith("#"):
                amount_hashtags += 1
                
            # Detect emoticons
            if word.startswith(":") and word.endswith(":"):
                amount_emoticons += 1
                
            # Check for flooding (3 of same characters in a row)
            for i in range(len(word)-2):
                if word[i] == word[i + 1] and word[i + 1] == word[i + 2]:
                    amount_flooding += 1
        
        # If numeric, save amount of flooding characters, else binary
        if amount_flooding > 0:
            if flooding_numeric is True:
                flooding.append(amount_flooding)
            else:
                flooding.append(1)
        else:
            flooding.append(0)
            
        # Calculate hashtag frequency ((amount of hashtags / tweet length in tokens) * 100)
        hashtag_freq.append((amount_hashtags / len(tweet)) * 100)
        
        # Calculate hashtag-to-word ratio (amount of hashtags / amount of non hashtag words)
        hashtag_to_word.append(division_nonzero(amount_hashtags, (len(tweet) - amount_hashtags)))
        
        # Calculate emoticon frequency ((amount of emoticons / tweet length in tokens) * 100)
        emoticon_freq.append((amount_emoticons / len(tweet)) * 100)
        
    # Add bags of n-grams to the feature set
    features.append(bag_of_words(token_unigrams, bow_length))
    features.append(bag_of_words(token_bigrams, bow_length))
    features.append(bag_of_words(char_trigrams_nosp, bow_length))
    features.append(bag_of_words(char_fourgrams_nosp, bow_length))
    
    # Add other features to the feature set
    features.append(punctuation)
    features.append(capitalisation)
    features.append(flooding)
    features.append(hashtag_freq)
    features.append(hashtag_to_word)
    features.append(emoticon_freq)
    features.append(tweet_length)

    return features

