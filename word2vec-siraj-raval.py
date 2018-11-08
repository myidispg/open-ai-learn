#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:37:58 2018

@author: myidispg
"""

from __future__ import absolute_import, division, print_function # Allows to use python 2 things in python 3
# for word encoding
import codecs
# regex
import glob
# concurrency
import multiprocessing
# dealing with os stuff
import os
# pretty printing, human readable
import pprint
# regular expressions
import re
# natural language toolkit
import nltk
# word2vec
import gensim.models.word2vec as w2v
# dimensionality reduction
import sklearn.manifold
# math
import numpy as np
# plotting
import matplotlib.pyplot as plt
# parse pandas as pd
import pandas
# visualisation
import seaborn as sns

# Step 1: Process our data
# Clean data
nltk.download('punkt') # Pretrained tokenizer
nltk.download('stopwords') # words like and, the, an, a, of.
# ^^ these words can be removed from word vectors and have little semantic meaning.

# Get the booknames, matching the text files.
book_filenames = sorted(glob.glob('GOT dataset/*.txt'))
print('Found books: ')
print(book_filenames)

# Combine the books into one string
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}' ...".format(book_filename.split('/')[1]))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now %d characters long!" % len(corpus_raw))
    print()
    
# Split the corpus into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

# Convert to list of words.
# Split into words, remove unnecessary words, no hyphens
# list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))