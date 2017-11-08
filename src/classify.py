import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from collections import Counter
from sklearn.linear_model import LogisticRegression
from pprint import pprint
# from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.preprocessing import normalize
from utils import *

best_lr = LogisticRegression(C=1000, class_weight='balanced', dual=False,
      fit_intercept=True, intercept_scaling=1, max_iter=100,
      multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
      solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

best_svm = LinearSVC(C=0.1, class_weight='balanced', dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

best_rf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=5)

def explain_svm(model,vocab):
    w = model.coef_[0]
    wv = sorted(zip(w, vocab))
    print 'top feats correlated to neg class (i.e., no WNV)'
    pprint(wv[:20])

    print 'top feats correlated to pos class (i.e., WNV)'
    pprint(sorted(wv[-20:], reverse=True))

def explain_rf(model,vocab):
    w = model.feature_importances_
    wv = sorted(zip(w, vocab))
    print 'top feats'
    pprint(sorted(wv[-40:], reverse=True))



def svm_fit (train,labels,vocab):
    acc = []
    p = []
    r = []
    f = []
    auc = []

    for i in xrange(5):


        X = train
        # X = normalize(X)
        # split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.3,random_state=randint(0,100))
        # X_train, X_test, y_train, y_test = train_test_split(train, labels,test_size=0.3,random_state=10)


        print 'X and y shapes/dist (before SMOTE): ', X_train.shape, Counter(y_train)
        # oversample using SMOTE
        # X_train, y_train = SMOTE(random_state=randint(0,100)).fit_sample(X_train, y_train)
        # print 'X_train and y_train shapes (after SMOTE): ', X_train.shape, Counter(y_train)

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
        acc.append(accuracy_score(y_test, y_pred))
        p.append(precision_score(y_test, y_pred))
        r.append(recall_score(y_test, y_pred))
        f.append(f1_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))
        print 'run: ', i + 1
        print classification_report(y_test, y_pred)
        explain_svm(best_model,vocab)

    mean_stds = print_mean_std(acc,p,r,f,auc)
    return mean_stds

def dt_fit (train,labels,vocab):

    acc = []
    p = []
    r = []
    f = []
    auc = []

    for i in xrange(5):
        #step: split dataset into train and test
        X = train#.as_matrix()
        # X = normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.3,random_state=randint(0,100))
        # X_train, X_test, y_train, y_test = train_test_split(train, labels,test_size=0.3,random_state=10)


        print 'X and y shapes/dist (before SMOTE): ', X_train.shape, Counter(y_train)
        # oversample using SMOTE
        # X_train, y_train = SMOTE(random_state=randint(0,100)).fit_sample(X_train, y_train)
        # print 'X_train and y_train shapes (after SMOTE): ', X_train.shape, Counter(y_train)

        #perform cv

        params = {'max_features':['auto','log2',None],
                  'class_weight':[None,'balanced']}
        clf = GridSearchCV(DecisionTreeClassifier(),params,n_jobs=-1,scoring='roc_auc',cv=3,verbose=2)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        export_graphviz(best_model,out_file = 'tree'+str(i+1)+'.dot',feature_names=vocab)
        print 'seleced best model: ', best_model

        #retrain best model
        best_model.fit(X_train,y_train)
        y_pred =  best_model.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        p.append(precision_score(y_test, y_pred))
        r.append(recall_score(y_test, y_pred))
        f.append(f1_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))
        print 'run: ', i + 1
        print classification_report(y_test, y_pred)
        explain_rf(best_model,vocab)

    mean_stds = print_mean_std(acc,p,r,f,auc)
    return mean_stds

if __name__ == '__main__':
    pass