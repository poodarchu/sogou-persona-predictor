# -*- coding=utf-8 -*-

import jieba
import jieba.posseg
import jieba.analyse
import gensim
import codecs
import time
import logging
from sklearn import feature_extraction
import string

class User(object):
    def __init__(self):
        self.userList = []
        self.userTags = []
        self.queryList = []


# split train.csv to user_list, tag_array and query_list.
def splitOriginal(filePath):

    userList = []
    userTag = []
    queryLists = []

    word2id = {}
    id2word = {}
    maxWID = 0

    with codecs.open(filePath, 'r', 'utf-8') as f:
        for line in f.readlines():
            line = line.split('\t')
            user, tag, queryList = line[0], [ line[1], line[2], line[3] ], line[4:]
            userList.append(user)
            userTag.append(tag)
            queryLists.append(queryList)
    f.close()

    print user

    stop_tokens = []
    fr = codecs.open('./data/stop_tokens.txt', 'r', 'utf-8')
    for token in fr.readlines():
        stop_tokens.append(token.strip())
    fr.close()

    queryLists_t = []
    with codecs.open('./data/output/train_token.csv', 'w', 'utf-8') as fw:
        for queryList in queryLists:
            qry = []
            final = ''
            for query in queryList:
                segs = jieba.cut(query, cut_all=False)
                for seg in segs:
                    if seg not in stop_tokens:
                        final += seg + ','
                        qry.extend(seg)
                final += '\n'
            fw.write(final)
            queryLists_t.append(qry)
        fw.close()

    queryLists = queryLists_t
    print len(queryLists)

    # 将得到的词语转换为词频矩阵
    freWord = feature_extraction.text.CountVectorizer()
    # 统计每个词语的tf-idf权值
    transformer = feature_extraction.text.TfidfTransformer()
    # 计算出tf-idf(第一个fit_transform),并将其转换为tf-idf矩阵(第二个fit_transformer)
    tfidf = transformer.fit_transform(freWord.fit_transform(queryLists))
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
                    tfidfDict.update({getWord: getValue})
    sorted_tfidf = sorted(tfidfDict.iteritems(), key=lambda d: d[1], reverse=True)
    fw = codecs.open('./data/output/tfidfDict', 'w', 'utf-8')
    for i in sorted_tfidf:
        fw.write(i[0] + '\t' + str(i[1]) + '\n')
    fw.close()


    # 当前循环所处理的语料子集，是一个 list 的 list，每个外层 list 对应一个用户的查询历史记录
    # 每个内层list为一串 (token_id, frequency) 的pair
    # 这是 gensim 的标准输入格式
    corpus = []

    # 保存原始信息
    orig_filename = './data/output/train.orig.txt'
    ORIG = codecs.open(orig_filename, 'w', 'utf-8')

    # 每行对应一个用户的查询，已经分好词
    for tokens in queryList:
        # 统计当前查询中每个词的频率
        qry_wid2freq = {}
        for token in tokens:
            ORIG.write(token)

            # 如果 token 已经在 word2id 中，映射成 wid
            if token in word2id:
                wid = word2id[token]
            else:
                wid = maxWID
                word2id[token] = maxWID
                id2word[wid] = token
                maxWID += 1

            # 统计 wid 的频率
            if wid in qry_wid2freq:
                qry_wid2freq[wid] += 1
            else:
                qry_wid2freq[wid] = 1
        ORIG.write('\n')

        # 将文章中的 wid 按 id 的大小排序
        sorted_wids = sorted(qry_wid2freq.keys())
        qry_pairs = []
        # 把 wid，frequency 的 pair 追加到当前 query 的 list 中
        for widid in sorted_wids:
            qry_pairs.append( (wid, qry_wid2freq[wid]) )

        # 当前文档的 list 已经全部生成，把它加入 subcorpus，即语料集的 list 中
        corpus.append(qry_pairs)

    ORIG.close()
    print "original queries saved in %s" % orig_filename

    print "Training LDA ..."
    startTime = time.time()
    lda = gensim.models.LdaModel(corpus=corpus, num_topics= 144, passes=1, update_every=100, iterations= 1000)
    endTime = time.time()
    print "Finished in %.1f seconds" %(endTime - startTime)

    lda.save('./data/output/lda.model')

    lda_fileName = './data/output/lda.txt'
    LDA = codecs.open(lda_fileName, 'w', 'utf-8')
    print "Saving topic propotrions into '%s' ... " %lda_fileName

    labels = userTag
    for d, doc_pairs in enumerate(corpus):
        lable = int(labels[d][0])
        # 将当前查询作为输入，用训练好的 LDA 模型，求 doc-topic 的比例
        topic_props = lda.get_document_topics(doc_pairs, minimum_probability=0.001)
        LDA.write('%d' %lable)
        # 把 k 个比例保存成 k 个特征 svmlight 格式
        for k, prop in topic_props:
            LDA.write(' %d:%.3f' %(k, prop))
        LDA.write('\n')
    LDA.close()

    print " %d docs saved." % len(corpus)

def calTfIdf():
    pass

if __name__ == '__main__':
    splitOriginal('./data/train.csv')
    print "split file to 3 then cut done."