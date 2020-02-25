# Information-Retrieval

# New Dataset 
https://drive.google.com/drive/folders/1pv98lYVuW26LHKoIVs8v_n7SVW7Kut-e

# Original Paper Dataset 
https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets

# Lexicons
## AFINN
https://pypi.org/project/afinn/
## GI, MPQA, NRC, LIU
https://github.com/beefoo/text-analysis


## TODO
### Preprocessing
1. Replace emoji by their name or description using Python emoji module
2. Normalize hyperlinks and @-replies
3. Tokenisation, PoS-tagging (Gimpel et al., 2011), lemmatization (Van de Kauter et al., 2013), NER (Ritter et al., 2011)

### Information sources
#### Lexical
4. bags-of-words
5. token unigrams, bigrams and character trigrams and fourgrams.
6. numeric and binary features containing info about character
7. punctuation flooding
8. punctuation
9. capitalisation
10. hashtag frequency
11. hashtag-to-word ratio
12. emoticon frequency
13. tweet length
(numerical features were normalised by dividing them by the tweet length in TOKENS (not characters))

#### Syntactic
##### For each PoS-tag
14. indicate whether it occurs in the tweet or not
15. whether the tag occurs 0, 1, or >2 times
16. frequency in absolute numbers
17. frequency as a percentage
##### Also
18. number of interjections
19. binary feature indicating a 'clash' between verb tenses (see Reyes et al. (2013))
##### For named entities
20. binary if there are named entities
21. number of named entities
22. number of tokens part of named entity
23. frequency of tokens part of named entity

#### Sentiment lexicon
24. number of positive, negative or neutral lexicon words averaged over text length (in tokens??)
25. overall tweet polarity (sum of values of the identified sentiment words)
26. difference between highest positive and lowest negative sentiment values
27. binary feature indicating whether there is a polarity contrast (at least one positive and one negative sentiment present)

#### Semantic
28. word embedding cluster features generated with Word2Vec (small, medium, large???), generated from separate background corpus of 45,251 English tweets, collected with #sarcasm, #irony and #not. Run Word2Vec on this corpus, applying the CBoW model, context size of 8, wordvec dimensionality of 200 features, cluster size of k = 2,000^3. Clusters were implemented as binary features, indicating for each cluster whether a word contained by that cluster occurs in the tweet.
