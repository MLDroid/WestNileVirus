from pprint import pprint
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC

def get_svm_feat_importances (df,labels,vocab):
    feat_names = vocab
    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(df.as_matrix(), labels)
    feat_imp_scores = lsvc.coef_[0]
    feat_imp_scores = sorted(zip(feat_imp_scores,feat_names),reverse=True)
    feat_imp_scores = [(sf[1],round(sf[0],6)) for sf in feat_imp_scores]
    pprint (feat_imp_scores)

def get_mi_feat_importances (df,labels):
    feat_names = list(df)
    feat_imp_scores = mutual_info_classif(df,labels).tolist()
    feat_imp_scores = sorted(zip(feat_imp_scores,feat_names),reverse=True)
    feat_imp_scores = [(sf[1],sf[0]) for sf in feat_imp_scores]
    pprint (feat_imp_scores)


if __name__ == '__main__':
    pass