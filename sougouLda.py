# -*- coding=utf-8 -*-

import gensim, logging, bz2
from gensim.models import ldamodel
import nltk
import codecs
import time
from gensim.corpora import Dictionary
import logging
import jieba

from sklearn import svm, metrics


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_data = []

def train():
    for line in codecs.open('./data/train.csv', 'r', 'utf-8').readlines():
        words = line.split('\t')
        train_data.append(words[4:])

    stopwords = codecs.open('./data/stop_tokens.txt', 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]

    train_set = []
    for line in train_data:
        for query in line:
            qs = list(jieba.cut(query))
            final = ''
            # for q in qs:
            #     if q not in stop_tokens:
            #         final += (seg + ',')
            # fw.write(final)
            train_set.append([w for w in qs if w not in stopwords])

    vocabulary = Dictionary(train_set)
    corpus = [ vocabulary.doc2bow(text) for text in train_set]


    print('Training LDA ... ')
    startTime = time.time()
    lda = ldamodel.LdaModel(corpus=corpus, id2word=vocabulary, num_topics=100, passes=2)
    endTime = time.time()
    print("Finished in %.1f seconds" %(endTime - startTime))
    print(lda.top_topics(100))

if __name__ == '__main__':
    train()