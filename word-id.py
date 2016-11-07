# -*- coding=utf-8

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import codecs
import numpy as np
import time
from multiprocessing import Process, Queue, Pool
import nltk
import itertools

import gensim
from gensim.models import LdaModel

# def calTfIdf(corpus, word_freq, word2index, index2id):
#     for k, v in word_freq.items:
#         tf = word_freq[k]
#         idf = np.log2(len(corpus)/)

# class WordIDUtil(object):
#     pass

corpus = []

def transfer():

    for line in codecs.open('tkd_qry_all.csv', 'r', 'utf-8').readlines():
        # print(line)
        words = line.split(',')
        corpus.append(words)

    # time.sleep(5)
    print('Corpus lenth: %d, words count: %d' %(len(corpus), 341663))

    print(type(corpus[10][12]))

    word_freq = nltk.FreqDist(itertools.chain(*corpus))
    # WordIDUtil.word_freq = word_freq
    print("Found %d unique words tokens." % len(word_freq.items()))

    # for k ,v in word_freq.items():
    #     print(k,v,'--------')

    vocabulary_size = len(word_freq)
    # WordIDUtil.vocabulary_size = vocabulary_size
    vocabulary = word_freq.most_common(vocabulary_size)
    # WordIDUtil.vocabulary = vocabulary

    index2word = [x[0] for x in vocabulary]
    # WordIDUtil.index2word = index2word
    word2index = dict([(w,i) for i, w in enumerate(index2word)])
    # WordIDUtil.word2index = word2index

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocabulary[-1][0], vocabulary[-1][1]))

    fw = codecs.open('term_idx_all.csv', 'w', 'utf-8')
    for line in corpus:
        for word in line:
            idx = word2index[word]
            fw.write(str(idx) + ' ')
        fw.write('\n')
    fw.close()

# word2id = {}
# id2word = []
# wordFreq = []
# maxIndex = 0
#
# for line in corpus:
#     for word in line:
#         if word not in id2word:
#             id2word.append(word)
#             word2id[word] = maxIndex
#             wordFreq.append(1)
#         else:
#             wordFreq[word2id[word]] += 1

# print(id2word[10] + '\t' + word2id[id2word[10]] + '\t' + wordFreq[10])
# print(len(id2word))
# print(len(word2id))
# print(len(wordFreq))

# # 将文本中的词语转换为词频矩阵 矩阵元素 a[i][j] 表示 j 词在 i 类文本下的词频
# vectorizer = CountVectorizer()
# # 该类会统计每个词语的 tf-idf 权值
# transformer = TfidfTransformer()
# # 第一个 fit_transform 是计算 tf-idf 第二个 fit_transform 是将文本转为词频矩阵
# tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# # 获取词袋模型中的所有词语
# words = vectorizer.get_feature_names()
# # 将 tf-idf 矩阵抽取出来，元素 w[i][j] 表示 j 词在 i 类文本中的 tf-idf 权重
# weight = tf_idf.toarray()
#
# res = 'tf-idf-result.csv'
# fw = codecs.open(res, 'w', 'utf-8')
#
# for j in range(len(words)):
#     fw.write(words[j] + ' ')
# fw.write('\n')
## fw.close()

if __name__ == '__main__':
    transfer()
