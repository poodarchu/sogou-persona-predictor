# -*- coding=utf-8 -*-

from sklearn import svm, metrics
from sklearn.datasets import load_svmlight_file
import sys
from sklearn.multiclass import  OneVsRestClassifier
import codecs

# 返回precision, recall, f1, accuracy
def getScores(true_classes, pred_classes, average):
    precision = metrics.precision_score(true_classes, pred_classes, average=average)
    recall = metrics.recall_score(true_classes, pred_classes, average=average)
    f1 = metrics.f1_score(true_classes, pred_classes, average=average)
    accuracy = metrics.accuracy_score(true_classes, pred_classes)
    return precision, recall, f1, accuracy



# 加载training和test文件的特征
for i in xrange(3):
    train_file = "train.svm-lda-%d.txt" % i
    test_file = "test.svm-lda-%d.txt" % i
    train_features_sparse, true_train_classes = load_svmlight_file(train_file)
    test_features_sparse, true_test_classes = load_svmlight_file(test_file)

    # 缺省加载为稀疏矩阵。转化为普通numpy array
    train_features = train_features_sparse.toarray()
    test_features = test_features_sparse.toarray()

    print "Train: %dx%d" % (train_features.shape)

    # 线性SVM，L1正则
    model = svm.LinearSVC(penalty='l1', dual=False)

    # 在training文件上训练
    print "Training... Predicting ...",
    model.fit(train_features, true_train_classes)
    print "Done."

    # 在test文件上做预测
    pred_train_classes = model.predict(train_features)
    pred_test_classes = model.predict(test_features)

    with codecs.open('./data/output/%d_predict.csv' % i, 'w', 'utf-8') as fw:
        for idx in pred_test_classes:
            fw.write('%d\n' % int(idx) )
        fw.close()


    # 汇报结果
    print metrics.classification_report(true_train_classes, pred_train_classes, digits=3)
    print metrics.classification_report(true_test_classes, pred_test_classes, digits=3)

    for average in ['micro', 'macro']:
        train_precision, train_recall, train_f1, train_acc = getScores(true_train_classes, pred_train_classes, average)
        print "Train Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" % (average, train_precision, train_recall, train_f1, train_acc)

        test_precision, test_recall, test_f1, test_acc = getScores(true_test_classes, pred_test_classes, average)
        print "Test Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" % (average, test_precision, test_recall, test_f1, test_acc)
