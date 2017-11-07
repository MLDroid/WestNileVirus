import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_cv (train,labels,n=5):
    acc = [];p = [];r = []; f = []
    for i in xrange(n):
        X_train, X_test, y_train, y_test = train_test_split(train, labels,test_size=0.3,random_state=randint(0,100))
        clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=5)
        clf.fit(X_train, y_train)
        y_pred =  clf.predict(X_test)
        acc.append(accuracy_score(y_test,y_pred))
        p.append(precision_score(y_test,y_pred))
        r.append(recall_score(y_test,y_pred))
        f.append(f1_score(y_test,y_pred))
        print 'run: ', i+1
        print classification_report(y_test,y_pred)

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