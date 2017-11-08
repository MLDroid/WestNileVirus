import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from explainable_svm import svm_explain
from utils import process_date, convert_species

import matplotlib.pyplot as plt


def barplot(feat_name, feat_categories, pos_values, neg_values):
    fig, ax = plt.subplots()

    index = np.arange(len(feat_categories))
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, pos_values, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Pos')

    rects2 = plt.bar(index + bar_width, neg_values, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Neg')

    plt.xlabel('feature categories')
    plt.ylabel('num samples')
    plt.title(feat_name)
    plt.xticks(index + bar_width / 2, feat_categories,rotation=90)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(feat_name+'.png')
    print 'check ',feat_name+'.png'

def visualize(train,labels):
    feat_names = list(train)
    for feat in feat_names:
        col = train[feat].values
        pos = Counter([col[i] for i,l in enumerate(labels) if l == 1])
        neg = Counter([col[i] for i,l in enumerate(labels) if l == 0])
        keys = list(set(pos.keys() + neg.keys()))
        keys.sort()
        pos_values = [pos.get(k,0) for k in keys]
        neg_values = [neg.get(k,0) for k in keys]
        barplot(feat,keys,pos_values,neg_values)


if __name__ == '__main__':
    pass