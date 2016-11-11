# -*- coding=utf-8 -*-

import codecs
import jieba

userID = []
# userTags = [] # userTag[i][0:3] : user i's three tags gender, age and certification
userQueries = [] # userQueries[i][:] user i's many queries
ages = []
genders = []
educations = []
with codecs.open('./data/train.csv', 'r', 'utf-8') as fr:
    for user in fr.readlines():
        userInfo = user.split('\t')
        # userTags.append([userInfo[1:4]])
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
    fw = codecs.open('./data/output/queries_tokenized.csv', 'w', 'utf-8')
    # fw_ages = codecs.open('./data/output/ages.csv', 'w', 'utf-8')
    # fw_genders = codecs.open('./data/output/genders.csv', 'w', 'utf-8')
    # fw_educations = codecs.open('./data/output/educations.csv', 'w', 'utf-8')

    for queriesPerUser in userQueries:
        queryList = []  # query list per user.
        for query in queriesPerUser:
            qry_tks = jieba.lcut(query, cut_all=False)
            final = ''
            for tk in qry_tks:
                if tk not in stop_tokens:
                    if tk != ' ':
                        queryList.append(tk)
                        final += tk + ','
            fw.write(final)
        fw.write('\n')
        queryLists.append(queryList)

    # for i in ages:
    #     fw_ages.write(i)
    #     fw_ages.write('\n')
    # for i in genders:
    #     fw_genders.write(i)
    #     fw_genders.write('\n')
    # for i in educations:
    #     fw_educations.write(i)
    #     fw_educations.write('\n')
    #
    # fw.close()
    # fw_ages.close()
    # fw_genders.close()
    # fw_educations.close()
    #
    # print userID[8]
    # print ages[8]
    # print genders[8]
    # print educations[8]
    # print queryLists[8][8]
    # print len(queryList[8])

    return userID, ages, genders, educations, queryLists

def cutTest2Rtn():
    fw = codecs.open('./data/output/test.csv', 'w', 'utf-8')
    testUIDs = []
    testQueryLists = []
    for queryPerLine in fw.readlines():
        queries = []
        userInfo = queryPerLine.split('\t')
        testUIDs.append(userInfo[0])

        for query in userInfo[1:]:
            qryTks = jieba.lcut(query)
            final = ''
            for i in qryTks:
                if i not in stop_tokens:
                    final += i + ','
                    queries.append(i)
            fw.write(final)
        testQueryLists.append(queries)

    return testUIDs, testQueryLists


if __name__ == '__main__':
    cut2rtn()
    cutTest2Rtn()
