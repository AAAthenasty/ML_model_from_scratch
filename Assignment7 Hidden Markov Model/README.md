In the programming section you will implement a named entity recognition system using Hidden Markov Models (HMMs). \
Named entity recognition (NER) is the task of classifying named entities, typically proper nouns, into pre-defined categories, s
uch as person, location, organization. Consider the example sequence below, where each word is appended with a tab and then its tag:

```
“ O 
Rhinestone B-ORG 
Cowboy I-ORG
” O 
( O 
Larry B-PER 
Weiss I-PER 
) O 
- O
3:15 O
```

Rhinestone and Cowboy are labeled as an organization (ORG), while Larry and Weiss is labeled as a person (PER). \
Words that aren’t named entities are assigned the O tag. \
The B- prefix indicates that a word is the beginning of an entity, \
while the I- prefix indicates that the word is inside the entity.

NER is an incredibly important task for a machine to begin to analyze and interpret a body of natural language text. \
For example, when designing a system that automatically summarizes news articles, it is important to recognize the key subjects in the articles. 
Another example is designing a trivia bot. If you can quickly extract the named entities from the trivia question, 
you may be able to more easily query your knowledge base (e.g. type a query into Google) to request information about the answer to the question.

# Dataset
The Dataset
WikiANN is a “silver standard” dataset that was generated without human labelling. The English Abstract Meaning Representation (AMR) corpus and DBpedia features were used to train an automatic classifier to label Wikipedia articles. \
These labels were then propagated throughout other Wikipedia articles using the Wikipedia’s cross-language links and redirect links. 

Afterwards, another tagger that self-trains on the existing tagged entities was used to label all other mentions of the same entities, 
even those with different morphologies (prefixes and suffixes that modify a word in other languages).

Finally, the amassed training examples were filtered by “commonness” and “topical relatedness” to pick more relevant training data.

The WikiANN dataset provides labelled entity data for Wikipedia articles in 282 languages. In this file, I only use the English subset, which contains 14,000 training examples and 3,300 test examples, 
and the French subset, which contains around 7,500 training examples and 300 test examples. 

On a technical level, the main task is to implement an algorithm to learn the HMM parameters given the training data \
and then implement the forward backward algorithm to perform a smoothing query which we can then use to predict the hidden tags for a sequence of words.
