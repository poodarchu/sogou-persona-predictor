# -*- coding=utf-8 -*-

import jieba

# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

# Transform a count matrix to a normalized tf or tf-idf representation
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import string
import codecs

#from numpy import *
fr = codecs.open('./data/train.csv')
fr_list = fr.read()
dataList = fr_list.split('\n')
fr.close()

data = []
for oneline in dataList:
	data.append(" ".join(jieba.cut(oneline)))

# 将得到的词语转换为词频矩阵
freWord = CountVectorizer()
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 计算出tf-idf(第一个fit_transform),并将其转换为tf-idf矩阵(第二个fit_transformer)
tfidf = transformer.fit_transform(freWord.fit_transform(data))
# 获取词袋模型中的所有词语
word = freWord.get_feature_names()
# 得到权重
weight = tfidf.toarray()
tfidfDict = {}
for i in range(len(weight)):
	for j in range(len(word)):
		getWord = word[j]
		getValue = weight[i][j]
		if getValue != 0:
			if tfidfDict.has_key(getWord):
				tfidfDict[getWord] += string.atof(getValue)
			else:
				tfidfDict.update({getWord:getValue})
sorted_tfidf = sorted(tfidfDict.iteritems(),
					  key = lambda d:d[1],reverse = True)
fw = codecs.open('./data/output/sex_split.csv','w', 'utf-8')
for i in sorted_tfidf:
	fw.write(i[0] + '\t' + str(i[1]) +'\n')
fw.close()
