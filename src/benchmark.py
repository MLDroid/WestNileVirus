import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import Counter
from explainable_svm import svm_explain
from utils import process_date, convert_species
from vis import visualize
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
from utils import to_categorical

from get_feat_importances import get_svm_feat_importances

MISSING_VAL = -1111


def split_lat_long_locations(lati,longi,num_lati_bins,num_longi_bins):
    step_size = (max(lati) - min(lati))/num_lati_bins
    lati_bins = np.arange(min(lati),max(lati),step=step_size).tolist() + [max(lati)+0.01]
    binned_lati =  [val-1 for val in np.digitize(lati,lati_bins).tolist()]

    step_size = (max(longi) - min(longi)) / num_longi_bins
    longi_bins = np.arange(min(longi), max(longi), step=step_size).tolist() + [max(longi)+0.01]
    binned_longi = [val-1 for val in np.digitize(longi, longi_bins).tolist()]

    locations = [lati_bin_num * num_lati_bins + longi_bin_num for lati_bin_num,longi_bin_num in zip(binned_lati,binned_longi)]
    return locations

def make_categorical_feats(df,feat_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[feat_name].values))
    categorical_feats = np.array(lbl.transform(train[feat_name].values))
    df[feat_name] = categorical_feats

    categorical_onehot_feats = to_categorical(categorical_feats)
    for dummy_dim in xrange(categorical_onehot_feats.shape[1]):
        df[feat_name+'_'+str(dummy_dim)] = categorical_onehot_feats[:,dummy_dim].tolist()


# Load dataset
def load_data():
    train = pd.read_csv('../input/train.csv')
    weather = pd.read_csv('../input/weather.csv')
    return train,weather

train,weather = load_data()

categorical_feats = ['Block','Street','Trap','Species']
feats_to_drop = ['Species', 'Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos',
                 'Date','Latitude','Longitude','Depth_x','AddressAccuracy'] + categorical_feats

# Get labels
labels = train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', MISSING_VAL)
weather = weather.replace('-', MISSING_VAL)
weather = weather.replace('T', MISSING_VAL)
weather = weather.replace(' T', MISSING_VAL)
weather = weather.replace('  T', MISSING_VAL)

process_date (train)
convert_species (train)

lati = train['Latitude'].values
longi = train['Longitude'].values
bin_sizes = [2,4,6,8,10]
for lati_bin in bin_sizes:
    for longi_bin in bin_sizes:
        locations = split_lat_long_locations(lati,longi,num_lati_bins = lati_bin, num_longi_bins = longi_bin)
        train['Loc_'+str(lati_bin)+'_'+str(longi_bin)] = locations

#normalize lat/long
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
train['Latitude_std'] = [i[0] for i in std_scaler.fit_transform(np.array(train['Latitude'].values).reshape(-1,1))]
train['Latitude_minmax'] = [i[0] for i in minmax_scaler.fit_transform(np.array(train['Latitude'].values).reshape(-1,1))]

train['Longitude_std'] = [i[0] for i in std_scaler.fit_transform(np.array(train['Longitude'].values).reshape(-1,1))]
train['Longitude_minmax'] = [i[0] for i in minmax_scaler.fit_transform(np.array(train['Longitude'].values).reshape(-1,1))]

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(lambda x: int(x*100))
train['Long_int'] = train.Longitude.apply(lambda x: int(x*100))


# Merge with weather data
train = train.merge(weather, on='Date')


# Convert categorical data to numbers
for feat in categorical_feats:
    make_categorical_feats(df=train, feat_name=feat)

#drop meaningless features
train = train.drop(feats_to_drop, axis = 1)

# drop columns with -1s
train = train.ix[:,(train != MISSING_VAL).any(axis=0)]

print 'dataset shape: ', train.shape
print 'label dist: ', Counter(labels)

# # Random Forest Classifier
# clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=5)
# clf.fit(train, labels)
#
# # create predictions and submission file
# predictions = clf.predict_proba(test)[:,1]
# sample['WnvPresent'] = predictions
# sample.to_csv('beat_the_benchmark.csv', index=False)


# visualize(train,labels)

# print run_cv(train,labels)

# print train
# raw_input()

vocab = list(train)
#std_scaler1 = StandardScaler()
#train = std_scaler1.fit_transform(train)
get_svm_feat_importances (train,labels,vocab)
svm_explain(train,labels,vocab)