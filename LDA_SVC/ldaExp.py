# -*- coding=utf-8 -*-

import time
import gensim
from corpusLoader import cut2rtn, cutTest2Rtn
import codecs

topicNum = 100

word2id = {}
id2word = {}
maxQID = 0


# 加载语料的train或test子集，单词以句子为单位放入 orig_docs_words，类比放在 orig_docs_cat
# setDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
# cats_docNames, category_names = loader(i)

UIDs, ages, genders, educations, queryLists = cut2rtn()
# testUIDs, testQueryLists = cutTest2Rtn()


# 当前循环所处理的语料子集，是一个list的list。每个外层list元素对应一个文档
# 每个内层list为一串 (word_id, frequency) 的pair
# 这种格式是gensim的标准输入格式
corpus = []

# 保存原始文本，以供人查看
orig_filename = "train.orig.txt"
ORIG = codecs.open(orig_filename, 'w', 'utf-8')
# 每个 wordsInSentences 对应一个文档
# 每个 wordsInSentences 由许多句子组成，每个句子是一个list of words
for queryList in queryLists:
    # 统计当前文档的每个词的频率，也就是说，每一篇文档都有一个word-frequency对照表。
    doc_tid2freq = {}
    # 循环取当前文档的一个句子
    for queryToken in queryList:

        ORIG.write("%s " % queryToken)

        # 如果w已在word2id映射表中，映射成wid
        if queryToken in word2id:
            qid = word2id[queryToken]
        # 否则，把w加入映射表，并映射成新wid
        else:
            qid = maxQID
            word2id[queryToken] = maxQID
            id2word[maxQID] = queryToken
            maxQID += 1

        # 统计 wid 的频率
        if qid in doc_tid2freq:
            doc_tid2freq[qid] += 1
        else:
            doc_tid2freq[qid] = 1

    ORIG.write("\n")
    # 把文档中出现的wid按id大小排序
    sorted_qids = sorted(doc_tid2freq.keys())
    doc_pairs = []
    # 把 (wid, frequency) 的对追加到当前文档的list中
    for qid in sorted_qids:
        doc_pairs.append((qid, doc_tid2freq[qid]))

    # 当前文档的list已经完全生成，把它加入subcorpus，即语料子集的list中
    corpus.append(doc_pairs)

ORIG.close()

print "Training LDA..."
startTime = time.time()
# LDA训练的时候是把train和test放一起训练的(更严格的办法应该是只用train集合来训练)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=topicNum, passes=2, iterations=200)
endTime = time.time()
print "Finished in %.1f seconds" % (endTime - startTime)

lda_filename = "train.svm-lda.txt"
LDA = codecs.open(lda_filename, 'w', 'utf-8')
print "Saving topic proportions into '%s'..." % lda_filename

# 拿出一个语料子集 (train或者test)
labels = ages

# 遍历子集中每个文档
for d, doc_pairs in enumerate(corpus):  # d is index, doc_pairs is the correspond value stored in corpus[d]
    label = int(labels[d])
    # 把当前文档作为输入，用训练好的LDA模型求“doc-topic比例”
    topic_props = lda.get_document_topics(doc_pairs, minimum_probability=0.001)
    LDA.write("%d" % label)
    # 把K个比例保存成K个特征，svmlight格式
    for k, prop in topic_props:
        LDA.write(" %d:%.3f" % (k, prop))
    LDA.write("\n")
LDA.close()

print "%d docs saved" % len(corpus)
