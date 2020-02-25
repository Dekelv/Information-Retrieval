# Information-Retrieval

# New Dataset 
https://drive.google.com/drive/folders/1pv98lYVuW26LHKoIVs8v_n7SVW7Kut-e

# Original Paper Dataset 
https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets


## TODO
### Preprocessing
1. Replace emoji by their name or description using Python emoji module
2. Normalize hyperlinks and @-replies
3. Tokenisation, PoS-tagging (Gimpel et al., 2011), lemmatization (Van de Kauter et al., 2013), NER (Ritter et al., 2011)

### Information sources
#### Lexical
2. bags-of-words
3. token unigrams, bigrams and character trigrams and fourgrams.
4. numeric and binary features containing info about character
5. punctuation flooding
6. punctuation
7. capitalisation
8. hashtag frequency
9. hashtag-to-word ratio
10. emoticon frequency
11. tweet length
(numerical features were normalised by dividing them by the tweet length in TOKENS (not characters))

#### Syntactic
For each PoS-tag
12. indicate whether it occurs in the tweet or not
13. whether the tag occurs 0, 1, or >2 times
14. frequency in absolute numbers
15. frequency as a percentage
Also
16. number of interjections
17. binary feature indicating a 'clash' between verb tenses (see Reyes et al. (2013))
For named entities
18. binary if there are named entities
19. number of named entities
20. number of tokens part of named entity
21. frequency of tokens part of named entity

#### Sentiment lexicon
22. number of positive, negative or neutral lexicon words averaged over text length (in tokens??)
23. overall tweet polarity (sum of values of the identified sentiment words)
24. difference between highest positive and lowest negative sentiment values
25. binary feature indicating whether there is a polarity contrast (at least one positive and one negative sentiment present)

#### Semantic
26. word embedding cluster features generated with Word2Vec (small, medium, large???), generated from separate background corpus of 45,251 English tweets, collected with #sarcasm, #irony and #not. Run Word2Vec on this corpus, applying the CBoW model, context size of 8, wordvec dimensionality of 200 features, cluster size of k = 2,000^3. Clusters were implemented as binary features, indicating for each cluster whether a word contained by that cluster occurs in the tweet.
