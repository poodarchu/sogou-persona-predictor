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
    with codecs.open('./data/output/0_predict.csv', 'r', 'utf-8') as fr:
        for age in fr:
            ages.append(int(age))
        fr.close

    genders = []
    with codecs.open('./data/output/1_predict.csv', 'r', 'utf-8') as fr:
        for gender in fr:
            genders.append(int(gender))
        fr.close

    educations = []
    with codecs.open('./data/output/2_predict.csv', 'r', 'utf-8') as fr:
        for edu in fr:
            educations.append(int(edu))
        fr.close

    with codecs.open('./data/output/UID_age_gender_education.csv', 'w', 'utf-8') as fw:
        uid_age = zip(UID, ages, genders, educations)
        for (uid, age, gender, education) in uid_age:
            fw.write('%s %s %d %d\n' % (uid, age, gender, education))
        fw.close()

