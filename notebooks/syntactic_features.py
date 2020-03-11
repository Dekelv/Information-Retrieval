#!/usr/bin/env python
# coding: utf-8

# ### Syntactic Features

# In[4]:


import spacy
nlp = spacy.load("en_core_web_sm")

def get_pos_and_ner(tweet):
    tweet = nlp(tweet)
    return ([(x.orth_, x.pos_,x.tag_, x.ent_type_) for x in [y for y in tweet if y.pos_ != 'SPACE']], tweet.ents) 


# In[58]:


from collections import Counter
from itertools import combinations

#Function that generates all syntactic features given a list of tweets.
def get_syntactic_features(tweets):
    syn_features = [];
    
    for tweet in tweets:
        #List of all possible coarse pos-tags. 'Space' pos-tag not included
        tagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        (pos_list,ents_list) = get_pos_and_ner(tweet)
        num_tokens = len(pos_list)
        pos_map = Counter([pos for (_,pos,_,_) in pos_list])
        
        pos_features = []
        for tag in tagset:
            
            #Generate the POS features
            bin_feat = 1 if pos_map[tag] > 0 else 0
            bound_freq_feat = 2 if pos_map[tag] > 1 else pos_map[tag]
            unbound_freq_feat = pos_map[tag]
            perc_feat = pos_map[tag] / num_tokens
            
            pos_features.append(bin_feat)
            pos_features.append(bound_freq_feat)
            pos_features.append(unbound_freq_feat)
            pos_features.append(perc_feat)
        
        #Fine POS tags that indicate verbs that are in past tense
        past_tense = ['VBD', 'VBN']
        #Fine POS tags that indicate verbs that are in present tense
        present_tense = ['VBG', 'VBP', 'VBZ', 'VB']
        fine_verb_tags_combs = [(tag1,tag2) for (tag1,tag2) in combinations([tag for (_,pos,tag,_) in pos_list if pos == 'VERB' or pos == 'AUX'],2)]
        clash_in_tense = [1 if (tag1 in past_tense and tag2 in present_tense) or (tag2 in past_tense and tag1 in present_tense) else 0 for (tag1,tag2) in fine_verb_tags_combs]
        clash_feature = 1 if 1 in clash_in_tense else 0
                 
        #Generate the numbered entity features
        num_ents = len(ents_list)
        bin_ents = 1 if num_ents > 0 else 0
        num_tokens_ents = len([ent_type for (_,_,_,ent_type) in pos_list if ent_type != ''])
            
        ent_features = [bin_ents, num_ents, num_tokens_ents]
        
        #Concatentate all syntactic subfeatures and append to list
        syn_features.append(pos_features + [clash_feature] + ent_features)
    
    return syn_features

