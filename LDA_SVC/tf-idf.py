# -*- coding=utf-8 -*-

import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import *
import string
import codecs

stdout = sys.stdout
reload(sys)
sys.stdout = stdout
print 1
sys.setdefaultencoding( "utf-8" )
sys.path.append("./")
#from numpy import *
fr = codecs.open('./data/train.csv')
fr_list = fr.read()
dataList = fr_list.split('\n')
data = []
for oneline in dataList:
	data.append(" ".join(jieba.cut(oneline)))

#将得到的词语转换为词频矩阵
freWord = CountVectorizer()
#统计每个词语的tf-idf权值
transformer = TfidfTransformer()
#计算出tf-idf(第一个fit_transform),并将其转换为tf-idf矩阵(第二个fit_transformer)
tfidf = transformer.fit_transform(freWord.fit_transform(data))
#获取词袋模型中的所有词语
word = freWord.get_feature_names()
#得到权重
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
fw = open(r'F:\study\master of TJU\DF\Sogou\DF-competition-sogou\data\sex\result_sex1_for_test.txt','w')
for i in sorted_tfidf:
	fw.write(i[0] + '\t' + str(i[1]) +'\n')