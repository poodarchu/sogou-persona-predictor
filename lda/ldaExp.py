# -*- coding=utf-8 -*-

import gensim
import time
from corpusLoader import *
import sys

# 从语料库名映射到加载函数，之后调用
corpus2loader = { '20news' : load_20news, 'reuters' : load_reuters }

def usage():
    print "Usage: ldaExp.py corpus_name"

corpusName = sys.argv[1]

#加载函数
loader = corpus2loader[corpusName]

# 20news 的文档数和类别数都多些，所以主题数设置大一些
if corpusName == '20news':
    topicNum = 100
else:
    topicNum = 50

# 两个语料库都已经分成 train 和 test 集合。后面分别处理
setNames = ['train', 'test']
baseNames = []
subCorpora = []
corpus = []
word2id = {}
id2word = {}
maxWID = 0

for setName in setNames:
    print "Process set '%s': " %setName

    # 加载语料库的train或者test子集，单词以句子为单位放入 origi_docs_words,类别放在 orig_docs_cat
    setDocNum, orig_docs_words, orig_docs_cat, cats_docsWords, cats_docNames, category_names = loader(setName)
    #文件名前缀
    baseName = "%s-%s-%d" %( corpusName, setName, setDocNum)
    baseNames.append(baseName)

    # 当前循环所处理的语料子集，是一个list的list。每个外层list元素对应一个文档
    # 每个内层list为一串 (word_id, frequency) 的pair
    # 这是 gensim 的标准输入格式
    subcorpus = []

    # 保存原始文本，以供人查看
    orig_filename = "./output/%s.orig.txt" %baseName
    ORIG = open(orig_filename, 'w')

    # 每个 wordsInSentences 对应一个文档
    # 每个 wordsInSentences 由许多句子组成，每个句子是一个 list of words
    for wordsInSentences in orig_docs_words:
        # 统计当前每个词的频率
        doc_wid2freq = {}
        # 循环读取当前文档的每个词的频率
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()
                ORIG.write("%s" %w)

                # 如果w已经在word2id映射表中，映射成wid
                if w in word2id:
                    wid = word2id[w]
                else:
                    wid = maxWID
                    word2id[w] = maxWID
                    id2word[wid] = w
                    maxWID += 1

                # 统计 wid 的频率
                if wid in doc_wid2freq:
                    doc_wid2freq[wid] += 1
                else:
                    doc_wid2freq[wid] = 1

        ORIG.write('\n')
        # 将文档中出现的 wid 按 id 大小排序
        sorted_wids = sorted(doc_wid2freq.keys())
        doc_pairs = []
        # 把(wid, frequency) 的pair 追加到当前文档的list中
        for wid in sorted_wids:
            doc_pairs.append( (wid, doc_wid2freq[wid]) )

        # 当前文档的 list 已经完全生成，把它加入 subcorpus，即语料集的list中
        subcorpus.append(doc_pairs)

    ORIG.close()
    print "%d original docs saced in %s" %( setDocNum, orig_filename )

    # 把整个语料子集list与之前的list合并，得到一个包含train和test集合的所有文档集合
    corpus += subcorpus
    # 这里把 train 和 test 集合分开放， 之后会把 不同集合的 每个文档的 "doc-topic比例" 保存成不同的文件
    subCorpora.append( (subcorpus, orig_docs_cat) )

print "Training LDA..."
stratTime = time.time()
# LDA 训练的时候是把train和test放在一起训练的，更严格的办法是只用train集合来训练
lda = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=topicNum, passes=20 )
endtime = time.time()
print "Finished in %.1f seconds" %(endtime - stratTime)

for i in xrange(2):
    lda_fileName = "./output/%s.svm-lda.txt" %baseNames[i]
    LDA = open( lda_fileName, 'w')
    print "Saving topic proportions into '%s' ... " %lda_fileName

    # 拿出一个语料子集（train 或者 test）
    subcorpus, labels = subCorpora[i]

    # 遍历子集中的每个文档
    for d, doc_pairs in enumerate(subcorpus):
        label = labels[d]
        # 将当前文档作为输入，用训练好的LDA模型求 doc-topic 的比例
        topic_props = lda.get_document_topics( doc_pairs, minimum_probability=0.001 )
        LDA.write( "%d" %label)
        # 把k个比例保存成k个特征 svmlight格式
        for k, prop in topic_props:
            LDA.write(" %d:%.3f" %(k, prop))
        LDA.write('\n')
    LDA.close()
    print "%d docs saved" %len(subcorpus)


