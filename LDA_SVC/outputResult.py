# -*- coding=utf-8 -*-

import codecs

if __name__ == '__main__':
    UID = []
    with codecs.open('./data/test.csv', 'r', 'utf-8') as fr:
        for user in fr.readlines():
            user = user.split('\t')
            UID.append(user[0])
        fr.close()

    ages = []
    with codecs.open('./data/output/age_predict.csv', 'r', 'utf-8') as fr:
        for age in fr:
            ages.append(int(age))
        fr.close

    with codecs.open('./data/output/UID_age.csv', 'w', 'utf-8') as fw:
        uid_age = zip(UID, ages)
        for (uid, age) in uid_age:
            fw.write('%s %s 0 0\n' % (uid, age))
        fw.close()