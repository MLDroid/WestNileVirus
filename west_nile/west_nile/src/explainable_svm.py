import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.preprocessing import normalize

best_lr = LogisticRegression(C=1000, class_weight='balanced', dual=False,
      fit_intercept=True, intercept_scaling=1, max_iter=100,
      multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
      solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

best_svm = LinearSVC(C=0.1, class_weight='balanced', dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

def explain(best_model,vocab):
    w = best_model.coef_[0]
    wv = sorted(zip(w,vocab))
    print 'top feats correlated to neg class (i.e., no WNV)'
    pprint(wv[:10])

    print 'top feats correlated to pos class (i.e., WNV)'
    pprint(sorted(wv[-10:],reverse=True))


def svm_explain (train,labels,vocab):
    print 'X and Y shapes (before SMOTE): ', train.shape, Counter(labels)
    print 'vocab len: ', len(vocab)
    print 'vocab'
    pprint (vocab)

    acc = [];
    p = [];
    r = [];
    f = []

    for i in xrange(5):

        #step: split dataset into train and test
        X = train#.as_matrix()
        # X = normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.3,random_state=randint(0,100))
        # X_train, X_test, y_train, y_test = train_test_split(train, labels,test_size=0.3,random_state=10)

        #oversample using SMOTE
        print 'X_train and y_train shapes (before SMOTE): ', X_train.shape, Counter(y_train)
        # X_train, y_train = SMOTE(random_state=42).fit_sample(X_train, y_train)
        # print 'X_train and y_train shapes (after SMOTE): ', X_train.shape, Counter(y_train)
        # raw_input()

        #perform cv
        # params = {'C':[0.01,0.1,1,10,100]}
        # clf = GridSearchCV(LinearSVC(class_weight='balanced',dual=False), params,n_jobs=-1,scoring='roc_auc',cv=3,verbose=2)
        # clf = GridSearchCV(LogisticRegression(class_weight='balanced'), params,n_jobs=-1,scoring='roc_auc',cv=5,verbose=2)
        # clf.fit(X_train, y_train)
        # best_model = clf.best_estimator_
        best_model = best_svm
        print 'seleced best model: ', best_model

        #retrain best model
        best_model.fit(X_train,y_train)
        y_pred =  best_model.predict(X_test)
        print accuracy_score(y_test,y_pred)
        print classification_report(y_test,y_pred)

        explain(best_model,vocab)

    acc = np.array(acc)
    p = np.array(p)
    r = np.array(r)
    f = np.array(f)
    acc_mean = acc.mean()
    acc_std = acc.std()
    p_mean = p.mean()
    p_std = p.std()
    r_mean = r.mean()
    r_std = r.std()
    f_mean = f.mean()
    f_std = f.std()
    return acc_mean, acc_std, p_mean, p_std, r_mean, r_std, f_mean, f_std

if __name__ == '__main__':
    pass