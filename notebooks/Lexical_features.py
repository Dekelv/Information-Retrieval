#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import re 
import heapq 
import string

from tqdm import tqdm_notebook
from nltk import ngrams
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


# In[89]:


def division_nonzero(n, d):
    return n / d if d else 0

## Get array of all lexical features
# If you want numeric flooding or punctuation features, make sure to set these parameters to True when calling the method
# bow_length is the length of your bag_of_words features.
def get_lexical_features(corpus, use_spaces=True, flooding_numeric=False, punctuation_numeric=False, bow_length=None):
    features = []
    
    # Token unigrams
    token_unigrams = corpus
    
    # Token bigrams
    token_bigrams = []
    
    # Character trigrams
    char_trigrams = []
    
    # Character fourgrams
    char_fourgrams = []    
    
    # Punctuation (numerical/binary)
    punctuation = []
    
    # Capitalisation (numerical/binary)
    capitalisation = []
    
    # Flooding (numerical/binary)
    character_flooding = []
    punctuation_flooding = []
    
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
    
    
    # Create all the n-grams
    for tweet in tqdm_notebook(corpus):
        # Create sentences, with and without spaces
        sentence = " ".join(tweet)
        sentence_nospace = "".join(tweet)
        
        # Token bigrams
        token_bigrams.append(list(ngrams(tweet, 2)))
        
        # Character trigrams and fourgrams
        if use_spaces is True:
            char_trigrams.append(list(ngrams(sentence, 3)))
            char_fourgrams.append(list(ngrams(sentence, 4)))
        else:
            char_trigrams.append(list(ngrams(sentence_nospace, 3)))
            char_fourgrams.append(list(ngrams(sentence_nospace, 4)))
            
    
    # Create all the bags of n-grams
    bag_of_unigrams = bag_of_words(token_unigrams, bow_length)
    bag_of_bigrams = bag_of_words(token_bigrams, bow_length)
    bag_of_trigrams = bag_of_words(char_trigrams, bow_length)
    bag_of_fourgrams = bag_of_words(char_fourgrams, bow_length)
    
    counter = 0
    
    # For every text in the corpus
    for tweet in tqdm_notebook(corpus):
        feature = []
        
        # Add bag of word entries for tweet
        feature = [*feature, *bag_of_unigrams[counter]]
        feature = [*feature, *bag_of_bigrams[counter]]
        feature = [*feature, *bag_of_trigrams[counter]]
        feature = [*feature, *bag_of_fourgrams[counter]]
        # Save the length in tokens
        tweet_length.append(len(tweet))
        
        # Create sentences, with and without spaces
        sentence = " ".join(tweet)
        sentence_nospace = "".join(tweet)

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
        amount_char_flooding = 0
        amount_punct_flooding = 0
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
                
            # Check for character flooding (3 of same characters in a row)
            for i in range(len(word)-2):
                if word[i] == word[i + 1] and word[i + 1] == word[i + 2] and word[i] not in string.punctuation:
                    amount_char_flooding += 1
            
        # Check for punctuation flooding (2 of same punctuation in a row)        
        for i in range(len(sentence_nospace)-1):
            if sentence_nospace[i] == sentence_nospace[i + 1] and sentence_nospace[i] in string.punctuation:
                amount_punct_flooding += 1
        
        # If numeric, save amount of flooding characters, else binary
        if amount_char_flooding > 0:
            if flooding_numeric is True:
                character_flooding.append(amount_char_flooding)
            else:
                character_flooding.append(1)
        else:
            character_flooding.append(0)
            
        # If numeric, save amount of flooding punctuation, else binary
        if amount_punct_flooding > 0:
            if flooding_numeric is True:
                punctuation_flooding.append(amount_punct_flooding)
            else:
                punctuation_flooding.append(1)
        else:
            punctuation_flooding.append(0)
            
        # Calculate hashtag frequency ((amount of hashtags / tweet length in tokens) * 100)
        hashtag_freq.append((amount_hashtags / len(tweet)) * 100)
        
        # Calculate hashtag-to-word ratio (amount of hashtags / amount of non hashtag words)
        hashtag_to_word.append(division_nonzero(amount_hashtags, (len(tweet) - amount_hashtags)))
        
        # Calculate emoticon frequency ((amount of emoticons / tweet length in tokens) * 100)
        emoticon_freq.append((amount_emoticons / len(tweet)) * 100)


        # Add other features to the feature set
        feature += punctuation
        feature += capitalisation
        feature += character_flooding
        feature += punctuation_flooding
        feature += hashtag_freq
        feature += hashtag_to_word
        feature += emoticon_freq
        feature += tweet_length
        
        counter += 1
        
        features.append(feature)

    return features

def bag_of_words(array, bow_length=None):
    # Map for word frequencies
    word2count = {} 

    # For every tweet, update word count for each word
    for tweet in array: 
        for word in tweet: 
            if word not in word2count.keys(): 
                word2count[word] = 1
            else: 
                word2count[word] += 1

    if bow_length is None:
        bow_length = len(word2count)
    freq_words = heapq.nlargest(bow_length, word2count, key=word2count.get)

    # Array for bags of words
    X = [] 
    for tweet in array: 
        vector = [] 
        for word in freq_words: 
            if word in tweet: 
                vector.append(1) 
            else: 
                vector.append(0) 
        X.append(vector) 
    return np.asarray(X)

