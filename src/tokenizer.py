# -*- coding=utf-8 -*-

import codecs
import jieba
import jieba.posseg
import jieba.analyse
import re

userList = []
userTag = []
queryLists = []


def chnTokeninzer(filePath):

    # jieba.analyse.set_stop_words("./data/stop_tokens.txt")

    with codecs.open(filePath, 'r', 'utf-8') as f:
        for line in f.readlines():
            line = line.split('\t')
            user, tag, queryList = line[0], (line[1], line[2], line[3]), line[4:]
            userList.append(user)
            userTag.append(tag)
            queryLists.append(queryList)
    f.close()

    print('-------我是华丽的分割线----------')
    print(len(queryLists))
    print(queryLists[0][0])
    seg_list = jieba.cut(queryLists[0][0], cut_all=False)  # seg_list is a generator
    print(type(seg_list))
    print("Default Mode: " + "/ ".join(seg_list))
    print('-------我是华丽的分割线----------')

    stop_tokens = []
    fr = codecs.open('./data/stop_tokens.txt', 'rb', 'utf-8')
    for token in fr.readlines():
        stop_tokens.append(token.strip())
    fr.close()
    # stop_tokens = {}.fromkeys(stop_tokens)
    # for k in stop_tokens[100:110]:
    #     print(k)

    with codecs.open('./output/tkd_qry_all.csv', 'w', 'utf-8') as fw:
        for queryList in queryLists:
            for query in queryList:
                seg_qry = jieba.cut(query, cut_all=False)
                final = ''
                for seg in seg_qry:
                        if seg not in stop_tokens:
                            final += (seg + ',')
                # if seg_qry[-1] not in stop_tokens:
                #     final += seg_qry[-1]
                fw.write(final)
            # fw.write('\n')
        fw.close()


if __name__ == '__main__':
    chnTokeninzer('./data/train.csv')
    print('Tokenization Done!')
