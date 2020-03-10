#!/usr/bin/env python
# coding: utf-8

# ### Preprocessing

# In[129]:


import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer
from IPython.display import display
from tqdm import tqdm_notebook


# In[5]:


# uncomment the following to download required libs
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('tagsets')


# In[6]:


df = pd.read_csv('data/original.txt', sep='\t', quotechar='~')


# In[7]:


tweet_labels = np.array(df['Label'])
tweets = np.array(df['Tweet text'])


# In[8]:


lemmatizer = WordNetLemmatizer() 
tweets_tokenized = [casual_tokenize(tweet) for tweet in tweets]
tweets_lemmatized = [[lemmatizer.lemmatize(token) for token in tweet] for tweet in tweets_tokenized]


# ### Name Entity Recognition (using SpaCy)

# In[18]:


import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[19]:


# only need to apply nlp once
# the entire background pipeline will return the objects
example = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
doc = nlp(example)
pprint([(X.text, X.label_) for X in doc.ents])


# In[21]:


# get number of extracted entities
print(f'number of entities: {len(doc.ents)}')


# In[22]:


# get entity labels and number
labels = [x.label_ for x in doc.ents]
Counter(labels)


# In[23]:


# displacy.render(nlp(example), jupyter=True, style='ent')


# In[24]:


# displacy.render(nlp(example), style='dep', jupyter = True, options = {'distance': 120})


# In[25]:


# we verbatim extract part-of-speech and lemmatize this sentence
[(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(example)) if not y.is_stop and y.pos_ != 'PUNCT']]


# ### Sentiment Features

# #### Affin scores

# In[34]:


df_affin = pd.read_csv('lexicons/afinn.txt', sep='\t')


# In[35]:


affin_map = dict()
affin_scores = list()
for index, row in df_affin.iterrows():
    # print(f'{row["word"]} {row["score"]}')
    affin_map[row["word"]] = row["score"]
    affin_scores.append(row["score"])
    
affin_min = min(affin_scores)
affin_max = max(affin_scores)


# In[36]:


# get the normalized affin score (range [-1, 1])
def get_affin_score(word):
    if affin_map.get(word) is not None:
        score = affin_map[word]
        return 2 * ((score - affin_min)/(affin_max - affin_min)) - 1
    else:
        return 0


# #### General Inquirer scores

# In[38]:


raw_data = pd.read_excel ('lexicons/inquirerbasic.xls')
raw_data = raw_data.as_matrix()


# In[39]:


gi_map = dict()
gi_scores = list()
for row in raw_data:
    word = row[0]
    positive = row[2]
    negative = row[3]
    score = 0
    if positive == "Positiv":
        score = 1
    elif negative == "Negativ":
        score = -1
    if word is not True and word is not False:
        gi_map[word.lower()] = score
        gi_scores.append(score)


# In[40]:


def get_gi_score(word):
    if gi_map.get(word) is not None:
        return gi_map.get(word)
    else:
        return 0


# #### MPQA scores

# In[42]:


df_mpqa = pd.read_csv('lexicons/MPQA.txt')


# In[43]:


mpqa_map = dict()
mpqa_scores = list()
for index, row in df_mpqa.iterrows():
    splits = row[0].split(' ')
    words = splits[2].split('=')
    scores = splits[len(splits)-1].split('=')
    score = 0
    if scores[1] == 'positive':
        score = 1
    elif scores[1] == 'negative':
        score = -1
    mpqa_map[words[1]] = score
    mpqa_scores.append(score)


# In[44]:


def get_mpqa_score(word):
    if mpqa_map.get(word) is not None:
        return mpqa_map.get(word)
    else:
        return 0


# #### Liu’s scores

# In[31]:


df_liu_pos = pd.read_csv('lexicons/liu-positive-words.txt')
df_liu_neg = pd.read_csv('lexicons/liu-negative-words.txt')


# In[46]:


liu_map = dict()
liu_scores = list()
for index, row in df_liu_pos.iterrows():
    liu_map[row["words"]] = 1
    liu_scores.append(1)
for index, row in df_liu_neg.iterrows():
    liu_map[row["words"]] = -1
    liu_scores.append(1)


# In[47]:


def get_liu_score(word):
    if liu_map.get(word) is not None:
        return liu_map.get(word)
    else:
        return 0


# #### NRC Emotion Lexicon

# In[32]:


df_nrc = pd.read_csv('lexicons/NRC.txt', sep='\t')


# In[49]:


nrc_map = dict()
nrc_scores = list()
item = 0
pos_score = None
neg_score = None
done = False
for index, row in df_nrc.iterrows():
    word = row["word"]
    if item == 10:
        item = 0
        pos_score = None
        neg_score = None
        done = False
    if item == 5:
        neg_score = row["score"]
    if item == 6:
        pos_score = row["score"]
    if pos_score is not None and neg_score is not None and not done:
        if pos_score != 0 and neg_score == 0:
            nrc_map[word] = 1
        elif pos_score == 0 and neg_score != 0:
            nrc_map[word] = -1
        else:
            nrc_map[word] = 0
        done = True
    item += 1


# In[50]:


def get_nrc_score(word):
    if nrc_map.get(word) is not None:
        return nrc_map.get(word)
    else:
        return 0


# #### Feature Extraction

# In[127]:


def get_features(document, lexicon):
    features = list()
    num_pos = 0
    num_neg = 0
    length = len(document)
    polarity = 0
    maximum = -1
    minimum = 1
    # contrast is a binary feature (i.e: zero means no contrast)
    contrast = 0
    for item in document:
        if lexicon == 'affin':
            score = get_affin_score(item)
        elif lexicon == 'gi':
            score = get_gi_score(item)
        elif lexicon == 'mpqa':
            score = get_mpqa_score(item)
        elif lexicon == 'liu':
            score = get_liu_score(item)
        else:
            score = get_nrc_score(item)
        polarity += score
        maximum = max(maximum, score)
        minimum = min(minimum, score)
        if score > 0:
            num_pos += 1
        elif score < 0:
            num_neg += 1  
    if num_pos > 0 and num_neg > 0:
        contrast = 1
    features.append(num_pos / length)
    features.append(num_neg / length)
    features.append((length - num_pos - num_neg) / length)
    features.append(polarity)
    features.append(maximum - minimum)
    features.append(contrast)
    return features


# In[133]:


def get_sentiment_features(corpus):
    features = list()
    for item in tqdm_notebook(corpus):
        feature = list()
        feature += get_features(item, 'affin')
        feature += get_features(item, 'gi')
        feature += get_features(item, 'mpqa')
        feature += get_features(item, 'liu')
        feature += get_features(item, 'nrc')
        features.append(feature)
    return features

