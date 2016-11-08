# -*- coding: utf-8 -*-

from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import codecs
import logging
from lxml import etree


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


train = []

def load_data(filePath):
    global train

    train_set = []
    fr = codecs.open(filePath, 'r', 'utf-8')
    for line in fr.readlines():
        line = line.split(' ')
        train_set.append(line)
    fr.close()

    # 构建训练语料
    dictionary = Dictionary(train_set)
    print(type(dictionary))

    corpus = [ dictionary.doc2bow(text) for text in train_set]
    print(type(corpus))

    # lda模型训练
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    lda.print_topics(20)


if __name__ == '__main__':
    load_data('term_idx_all.csv')