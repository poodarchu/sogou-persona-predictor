# -*- coding=utf-8 -*-
import re
import random

def file2utf8():
    fr = open('./data/train.csv', 'rb')
    fw = open('./data/train_utf8.csv', 'w')
    for line in fr.readlines():
        line = line.decode('gb18030')
        fw.write(line.encode('utf-8'))
    fr.close()
    fw.close()

    frt = open('./data/test.csv', 'rb')
    fwt = open('./data/test_utf8.csv', 'w')
    for line in frt.readlines():
        line = line.decode("gb18030")
        fwt.write(line.encode('utf-8'))
    frt.close()
    fwt.close()

userList = []
queryList = []

def splitUserAndQuery(filePath):
    fr = open(filePath, 'rb')
    for line in fr.readlines():
        listPerLine = line.decode('utf-8').split('\t')
        userList.append(listPerLine[0:4])
        queryList.append(listPerLine[4:])

    f = open('./output/userList.csv', 'w')
    for user in userList:
        for item in user:
            f.write(item + '\t')
        f.write('\n')
    f.close()

    fq = open('./output/queryList.csv', 'w')
    for query in queryList:
        for item in query:
            item = item.encode('utf-8')
            if re.match('.*https?:.*', item):
                item = ''
            fq.write(item + '\t')
    fq.close()


def filterData(filePath, outputFilePath):
    fr = open(filePath, 'rb')
    fw = open(outputFilePath, 'w')

    for line in fr.readlines():
        listPerLine = line.decode('utf-8').split('\t')
        for item in listPerLine[:-2]:
            # if re.match('.*https?:.*', item):
            if re.match('.*https?:.*', item):
                item = ''
            else:
                fw.write(item.encode('utf-8') + '\t')
        item = listPerLine[-1]
        fw.write(item.encode('utf-8'))

    fr.close()
    fw.close()


def splitTrainSet(inputFilePath):
    fr = open(inputFilePath, 'rb')
    queryList = fr.readlines()
    random.shuffle(queryList)

    fw1 = open('./output/test_s.csv', 'w')
    fw2 = open('./output/train_s.csv', 'w')

    lenth = len(queryList)
    for line in queryList[:int(lenth*0.2)]:
        fw1.write(line)
    for line in queryList[int(lenth*0.2):]:
        fw2.write(line)

    fw1.close()
    fw2.close()

if __name__ == '__main__':
    # splitUserAndQuery('./data/train_utf8.csv')
    # filterData('./data/train_utf8.csv', './data/train_utf8_cleaned.csv')
    # filterData('./data/test_utf8.csv', './data/test_utf8_cleaned.csv')
    splitTrainSet('./data/train.csv')