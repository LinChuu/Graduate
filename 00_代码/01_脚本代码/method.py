#accuracybag
#import accuracy as ac
#method bag
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from svmutil import *
import datetime
import time
# from sklearn import multiclass

# Logistic Regression
def SVM_NO(imagelist, lablelist, test_imagelist, test_lablelist):
    #model
    # method name
    methodname = "Support Vecter Machine"
    # clf = svm.LinearSVC(C=1.0, class_weight=None,dual=True,fit_intercept=True,intercept_scaling=1,loss='squared_hinge',max_iter=1000,multi_class='crammer_singer',penalty='l2',random_state=0,tol=1e-05,verbose=0)
    clf = svm.SVC(C=1.0, class_weight=None, max_iter=-1, decision_function_shape='ovo',random_state=0,tol=1e-05, verbose=0)
    # clf = svm.SVC()
    clf.fit(imagelist, lablelist)
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA=ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA,methodname,F1SCORE,confusion
# 20200505 验证cache_size的实验
def SVM_p(imagelist, lablelist, test_imagelist, test_lablelist,G,C,cache_size):
    #model
    methodname = "Support Vecter Machine"
    # clf=svm.LinearSVC(C=C,max_iter=I,random_state=0,verbose=0)
    begin_time = datetime.datetime.now()
    #clf = svm.SVC(C=C, kernel='rbf', gamma=G, class_weight=None, tol=10,cache_size=cache_size)
    clf = svm.SVC(C=C,gamma=G, kernel='rbf',cache_size=cache_size)
    clf.fit(imagelist, lablelist)
    end = datetime.datetime.now()
    rtime = end - begin_time
    running = rtime.total_seconds()
    # support = np.shape(clf.support_)
    rtnumber_n_support=clf.n_support_
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA, methodname, F1SCORE, confusion, running, rtnumber_n_support
# Logistic Regression
def SVM_libsvm(imagelist, lablelist, test_imagelist, test_lablelist,G,C,cache_size):
    #model
    methodname = "Support Vecter Machine"
    # clf=svm.LinearSVC(C=C,max_iter=I,random_state=0,verbose=0)
    begin_time = time.strftime('%H''%M''%S', time.localtime(time.time()))
    begin_H = float(begin_time[0:2])
    begin_M = float(begin_time[2:4])
    begin_S = float(begin_time[4:6])
    clf = svm_train(imagelist,lablelist,'-c ')
    p_label, p_acc, p_val=svm_predict(test_imagelist, test_lablelist,clf)
    clf.fit(imagelist, lablelist)
    end_time = time.strftime('%H''%M''%S', time.localtime(time.time()))
    end_H = float(end_time[0:2])
    end_M = float(end_time[2:4])
    end_S = float(end_time[4:6])
    rtime = ((begin_H - end_H) * 60 + (begin_M - end_M)) * 60 + (begin_S - end_S)
    # support = np.shape(clf.support_)
    rtnumber_n_support=clf.n_support_
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA, methodname, F1SCORE, confusion, rtime, rtnumber_n_support
# Logistic Regression
def Logisticregression(imagelist, lablelist, test_imagelist, test_lablelist):
    # model
    clf = LogisticRegression(penalty= 'l2',solver='sag',multi_class='multinomial',C=10000.0, random_state=0,max_iter=100)
    clf.fit(imagelist, lablelist)
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    kappa = ac.kappa(pretest, test_lablelist)
    # method name
    methodname = 'Logistic Regression'
    # print
    return testaccuracy ,methodname,F1SCORE,kappa
def LR_p(imagelist, lablelist, test_imagelist, test_lablelist,I,C):
    # model
    methodname = 'Logistic Regression'
    # TIME
    begin = datetime.datetime.now()
    clf = LogisticRegression(penalty= 'l2',solver='sag',multi_class='multinomial',C=0.00001)
    clf.fit(imagelist, lablelist)
    end = datetime.datetime.now()
    rtime = end - begin
    running = rtime.total_seconds()
    iter= clf.n_iter_[0]
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA, methodname, F1SCORE, confusion, running,iter
# k-NearestNeighbor,kNN
def knearestneighbor(imagelist, lablelist, test_imagelist, test_lablelist):
    # model
    clf = KNeighborsClassifier(n_neighbors=5,weights='uniform')
    clf.fit(imagelist, lablelist)
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # method name
    methodname = 'k-NearestNeighbor'
    # print
    return testaccuracy,methodname,F1SCORE,confusion
# design for parameter choice
def kNN_p(imagelist, lablelist, test_imagelist, test_lablelist,k,weigh,p):
    # model
    # method name
    methodname = 'k-NearestNeighbor'
    # time count
    begin=datetime.datetime.now()
    clf = KNeighborsClassifier(n_neighbors=k,weights=weigh,p=p)
    clf.fit(imagelist, lablelist)
    end = datetime.datetime.now()
    rtime = end-begin
    running = rtime.total_seconds()
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy,running, KAPPA, methodname, F1SCORE, confusion
# RandomForest


def RF_p(imagelist, lablelist, test_imagelist, test_lablelist,S):
    # n_estimators,min_samples_split
    methodname = 'Random Forest'
    # model
    begin = datetime.datetime.now()
    clf = RandomForestClassifier(n_estimators=S)
    # clf = RandomForestClassifier(n_estimators=180, max_depth=None, min_samples_split=10)
    clf.fit(imagelist, lablelist)
    end = datetime.datetime.now()
    rtime = end - begin
    running = rtime.total_seconds()
    # predict image
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA, methodname, F1SCORE, confusion, running

# CART
def decisiontree(imagelist, lablelist, test_imagelist, test_lablelist):
    # model
    clf = DecisionTreeClassifier(max_depth=None, random_state=0,criterion='gini')
    clf.fit(imagelist, lablelist)
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # method name
    methodname = 'Desision Tree'
    # print
    return testaccuracy, methodname, F1SCORE, confusion
def CART_P(imagelist, lablelist, test_imagelist, test_lablelist,M):
    # model
    # method name
    methodname = 'Desision Tree'
    begin = datetime.datetime.now()
    clf = DecisionTreeClassifier(max_depth=None, random_state=0,criterion='gini',min_samples_split=M)
    clf.fit(imagelist, lablelist)
    end = datetime.datetime.now()
    rtime = end - begin
    running = rtime.total_seconds()
    # predict test1
    pretest = clf.predict(test_imagelist)
    testaccuracy = ac.accuracy(pretest, test_lablelist)
    F1SCORE = ac.f1(pretest, test_lablelist)
    KAPPA = ac.kappa(pretest, test_lablelist)
    confusion = ac.confusion(pretest, test_lablelist)
    # print
    return testaccuracy, KAPPA, methodname, F1SCORE, confusion, running

def randomforest(imagelist, lablelist, test_imagelist, test_lablelist):
    # n_estimators,min_samples_split
    # model
    clf = RandomForestClassifier(max_depth=None)
    # clf = RandomForestClassifier(n_estimators=180, max_depth=None, min_samples_split=10)
    clf.fit(imagelist, lablelist)
    # predict test1
    pretest = clf.predict(test_imagelist)
    #testaccuracy = ac.accuracy(pretest, test_lablelist)
    #F1SCORE = ac.f1(pretest, test_lablelist)
    #kappa = ac.kappa(pretest, test_lablelist)
    # a=sklearn.metrics.confusion_matrix(test_lablelist,pretest,labels=None,sample_weight=None)
    # print(a)
    # method name
    #methodname = 'Random Forest'
    # print
    #return testaccuracy, methodname, F1SCORE, kappa


if __name__ == '__main__':
    print(1111)
    #randomforest()




