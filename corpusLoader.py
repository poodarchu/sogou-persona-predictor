# -*- coding=utf-8 -*-

import codecs
import jieba

userID = []
userTags = [] # userTag[i][0:3] : user i's three tags gender, age and certification
userQueries = [] # userQueries[i][:] user i's many queries
ages = []
genders = []
educations = []
with codecs.open('./data/train.csv', 'r', 'utf-8') as fr:
    for user in fr.readlines():
        userInfo = user.strip().split('\t')
        userTags.append([userInfo[1:4]])
        userID.append(userInfo[0])
        ages.append(userInfo[1])
        genders.append(userInfo[2])
        educations.append(userInfo[3])
        userQueries.append(userInfo[4:])
    fr.close()

stop_tokens = []
fr = codecs.open('./data/stop_tokens.txt', 'r', 'utf-8')
for token in fr.readlines():
    stop_tokens.append(token.strip())
fr.close()

queryLists = []

def cut2rtn():

    for queriesPerUser in userQueries:
        queryList = []  # query list per user.
        for query in queriesPerUser:
            qry_tks = jieba.cut(query, cut_all=False)
            for tk in qry_tks:
                if tk not in stop_tokens:
                    queryList.append(tk)
        queryLists.append(queryList)

    return userID, ages, genders, educations, queryLists
