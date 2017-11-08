import numpy as np
import pandas as pd


# Load dataset
def load_data():
    train = pd.read_csv('../input/train.csv')
    weather = pd.read_csv('../input/weather.csv')
    return train,weather

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def print_mean_std (acc,p,r,f,auc):
    acc = np.array(acc)
    p = np.array(p)
    r = np.array(r)
    f = np.array(f)
    auc = np.array(auc)

    acc_mean = acc.mean()
    acc_std = acc.std()
    p_mean = p.mean()
    p_std = p.std()
    r_mean = r.mean()
    r_std = r.std()
    f_mean = f.mean()
    f_std = f.std()
    auc_mean = auc.mean()
    auc_std = auc.std()
    print 'acc:',acc_mean, acc_std
    print 'p/r/f:',p_mean,p_std,r_mean,r_std,f_mean,f_std
    print 'auc:',auc_mean,auc_std
    return acc_mean, acc_std, p_mean, p_std, r_mean, r_std, f_mean, f_std, auc_mean, auc_std




if __name__ == '__main__':
    pass