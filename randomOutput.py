# -*- coding=utf-8 -*-

import random

def randomResult(inputFilePath, outputFilePath):
    fr = open(inputFilePath, 'rb')
    fw = open(outputFilePath, 'w')

    userList = []
    for line in fr.readlines():
        list = line.split('\t')
        userList.append(list[0])

    # print len(userList)
    # for i in userList[0]



    # print age, gender, education
    for user in userList:
        age = random.randint(0, 6)
        gender = random.randint(0, 2)
        education = random.randint(0, 6)
        result = [str(user), str(age), str(gender), str(education)]

        # print result
        # print type(user)

        for i in result:
            i.decode('utf-8').encode('GBK')
            fw.write(i + ' ')
        fw.write('\n')

    fr.close()
    fw.close()

if __name__ == '__main__':
    randomResult('./data/test.csv', 'randomResult.csv')