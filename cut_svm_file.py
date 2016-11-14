# -*- coding=utf-8 -*-

import codecs

if __name__ == '__main__':
    ages = []
    genders = []
    educations = []
    topic_prob = []

    with codecs.open('train.svm-lda.txt', 'r', 'utf-8') as fr:
        for doc in fr.readlines():
            tk = doc.split(' ')
            ages.append(tk[0])
            genders.append(tk[1])
            educations.append(tk[2])
            topic_prob.append(tk[3:])
        fr.close()
    print len(ages), len(genders), len(educations), len(topic_prob)

    for i in xrange(3):
        with codecs.open('train.svm-lda-%d.txt' % i, 'w', 'utf-8') as fw:
            for idx in xrange(len(ages)):
                if i == 0:
                    fw.write('%d ' % int(ages[idx]) )
                elif i == 1:
                    fw.write('%d ' % int(genders[idx]))
                else:
                    fw.write('%d ' % int(educations[idx]))

                for idxx in xrange(len(topic_prob[idx])):
                    fw.write('%s ' % str(topic_prob[idx][idxx]))
        fw.close()

    # Process test file
    ages = []
    genders = []
    educations = []
    topic_prob = []

    with codecs.open('test.svm-lda.txt', 'r', 'utf-8') as fr:
        for doc in fr.readlines():
            tk = doc.split(' ')
            ages.append(tk[0])
            genders.append(tk[1])
            educations.append(tk[2])
            topic_prob.append(tk[3:])
        fr.close()
    print len(ages), len(genders), len(educations), len(topic_prob)

    for i in xrange(3):
        with codecs.open('test.svm-lda-%d.txt' % i, 'w', 'utf-8') as fw:
            for idx in xrange(len(ages)):
                if i == 0:
                    fw.write('%d ' % int(ages[idx]))
                elif i == 1:
                    fw.write('%d ' % int(genders[idx]))
                else:
                    fw.write('%d ' % int(educations[idx]))

                for idxx in xrange(len(topic_prob[idx])):
                    fw.write('%s ' % str(topic_prob[idx][idxx]))
        fw.close()