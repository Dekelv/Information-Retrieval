#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import os
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import pathlib

scores = {} # scores is an empty dict alreadydef __init__(self):
features_size=200
path = pathlib.Path(__file__).parent.absolute()
target = str(path) + '/knnModel.sav'
means = pickle.load(open(target,'rb'))
    


# In[2]:


from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load(str(path) + "/word2vec/word2vec.model")

wordvectors=model.wv
features_size=200

vectors = np.zeros((len(wordvectors.vocab),200))
words = {}
word_count=0
for i in range(len(wordvectors.vocab)):
    words[wordvectors.index2entity[i]]=i
    vectors[i]=wordvectors[wordvectors.index2entity[i]]


# In[3]:



##get_features takes a array of tokens(sentence) and returns a feature vector 
def get_features(sentence):
    features = np.zeros(features_size)
    for word in sentence:
        feature = means.predict(vectors[words[word]].reshape(1, 200))
        features[feature]=1
    return features


# In[4]:


get_features(['hello','hi'])


# In[ ]:




