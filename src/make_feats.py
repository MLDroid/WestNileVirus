import pandas as pd
import numpy as np
from sklearn import preprocessing

from utils import to_categorical
from pprint import pprint

MISSING_VAL = -1111

#######################################################################################################################
#           categorical to numeric (dummy) variable conversion
#######################################################################################################################
def make_categorical_feats(df,feat_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[feat_name].values))
    categorical_feats = np.array(lbl.transform(df[feat_name].values))
    df[feat_name] = categorical_feats

    categorical_onehot_feats = to_categorical(categorical_feats)
    for dummy_dim in xrange(categorical_onehot_feats.shape[1]):
        df[feat_name+'_'+str(dummy_dim)] = categorical_onehot_feats[:,dummy_dim].tolist()


#######################################################################################################################
#           location specific features
#######################################################################################################################
def split_lat_long_locations(lati,longi,num_lati_bins,num_longi_bins):
    step_size = (max(lati) - min(lati))/num_lati_bins
    lati_bins = np.arange(min(lati),max(lati),step=step_size).tolist() + [max(lati)+0.01]
    binned_lati =  [val-1 for val in np.digitize(lati,lati_bins).tolist()]

    step_size = (max(longi) - min(longi)) / num_longi_bins
    longi_bins = np.arange(min(longi), max(longi), step=step_size).tolist() + [max(longi)+0.01]
    binned_longi = [val-1 for val in np.digitize(longi, longi_bins).tolist()]

    locations = [lati_bin_num * num_lati_bins + longi_bin_num for lati_bin_num,longi_bin_num in zip(binned_lati,binned_longi)]
    return locations


#######################################################################################################################
#           weather features
#######################################################################################################################
def merge_weather_stations(weather):
    #some cleanup
    weather = weather.drop('CodeSum', axis=1)
    weather.PrecipTotal = weather.PrecipTotal.str.strip()

    weather_stn1 = weather[weather['Station'] == 1]
    weather_stn2 = weather[weather['Station'] == 2]
    weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn2 = weather_stn2.drop('Station', axis=1)
    weather = weather_stn1.merge(weather_stn2, on='Date')
    return weather


def replace_missing_weather_values (weather):
    # replace some missing values and T with -1
    weather = weather.replace('M', MISSING_VAL)
    weather = weather.replace('-', MISSING_VAL)
    weather = weather.replace('T', MISSING_VAL)
    weather = weather.replace(' T', MISSING_VAL)
    weather = weather.replace('  T', MISSING_VAL)
    return weather

def make_day_night_length_feats(weather):
    weather.Sunset_x = weather.Sunset_x.replace('\+?60', '59', regex=True)
    time_func = lambda x: pd.Timestamp(pd.to_datetime(x, format='%H%M'))
    Sunrise_x_time = weather.Sunrise_x.apply(time_func)
    Sunset_x_time = weather.Sunset_x.apply(time_func)
    weather['DayLength_MPrec'] = (Sunset_x_time - Sunrise_x_time).astype('timedelta64[m]') / 60
    weather['DayLength_NearH'] = np.round(((Sunset_x_time - Sunrise_x_time).astype('timedelta64[m]') / 60).values)
    weather['NightLength_MPrec'] = 24.0 - weather.DayLength_MPrec
    weather['NightLength_NearH'] = 24.0 - weather.DayLength_NearH
    return weather


#######################################################################################################################
#           day/time features
#######################################################################################################################
def process_date(df):
    def create_year(x):
        return int(x.split('-')[0])

    def create_month(x):
        return int(x.split('-')[1])

    def create_day_month(x):
        return int(x.split('-')[2])

    def create_day_year(x):
        month = int(x.split('-')[1])-1
        day_month = int(x.split('-')[2])
        return month*30 + day_month

    def create_week(x):
        month = int(x.split('-')[1]) - 1
        day_month = int(x.split('-')[2])
        month_to_weeks = month * 4
        day_to_weeks = day_month/7 + 1
        week = month_to_weeks + day_to_weeks
        return int(week)



    df['year'] = df.Date.apply(create_year)
    df['month'] = df.Date.apply(create_month)
    df['day_of_month'] = df.Date.apply(create_day_month)
    df['day_of_year'] = df.Date.apply(create_day_year)
    df['week_of_year'] = df.Date.apply(create_week)
    return df

def convert_species(df):
    df['IsPipiens'] = ((df['Species']=='CULEX PIPIENS'  )*1 + (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    df['IsPipiensRestuans'] = ((df['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                               (df['Species']=='CULEX PIPIENS')*0 + (df['Species']=='CULEX RESTUANS'  )*0)
    df['IsRestuans'] = ((df['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                               (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    df['IsOther'] = (df['Species']!='CULEX PIPIENS') * (df['Species']!='CULEX PIPIENS/RESTUANS') * (df['Species']!='CULEX RESTUANS')


if __name__ == '__main__':
    pass