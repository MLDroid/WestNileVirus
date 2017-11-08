import pandas as pd
import numpy as np
from collections import Counter
from classify import *

from vis import visualize
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from make_feats import *
from utils import *
from get_feat_importances import get_svm_feat_importances

MISSING_VAL = -1111
categorical_feats = ['Block','Street','Trap','Species']
feats_to_drop = ['Species', 'Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos',
                 'Date','Latitude','Longitude','Depth_x','AddressAccuracy'] + categorical_feats




train,weather = load_data()

# Get labels
labels = train.WnvPresent.values

# Split station 1 and 2 and join by date
weather = merge_weather_stations(weather)
weather = replace_missing_weather_values(weather)
weather = make_day_night_length_feats(weather)

#make new features
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
train['Latitude_std'] = [i[0] for i in std_scaler.fit_transform(np.array(train['Latitude'].values).reshape(-1,1))]
train['Longitude_std'] = [i[0] for i in std_scaler.fit_transform(np.array(train['Longitude'].values).reshape(-1,1))]

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

# drop samples with MISSING_VAL
train = train.ix[:,(train != MISSING_VAL).any(axis=0)]
print 'dataset shape: ', train.shape
print 'label dist: ', Counter(labels)


# visualize(train,labels)



vocab = list(train)
get_svm_feat_importances (train,labels,vocab)
svm_fit(train,labels,vocab)
# dt_fit(train,labels,vocab)