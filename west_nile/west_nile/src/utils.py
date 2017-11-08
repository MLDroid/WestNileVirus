import numpy as np

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def process_date(df):
    '''
    Extract the year, month and day from the date
    '''

    # Functions to extract year, month and day from dataset
    def create_year(x):
        return int(x.split('-')[0])

    def create_month(x):
        return int(x.split('-')[1])

    def create_day(x):
        return int(x.split('-')[2])

    df['year'] = df.Date.apply(create_year)
    df['month'] = df.Date.apply(create_month)
    df['day'] = df.Date.apply(create_day)

def convert_species(df):
    '''
    Convert the Species field into 4 attributes: IsPipiens, IsPipiensRestuans (gets 0.5 for Pipiens and for Restuans),
    IsRestuans, and IsOther (for all other species).
    '''
        # % Wnv  / total count in train data
    df['IsPipiens'] = ((df['Species']=='CULEX PIPIENS'  )*1 + (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    df['IsPipiensRestuans'] = ((df['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                               (df['Species']=='CULEX PIPIENS')*0 + (df['Species']=='CULEX RESTUANS'  )*0)
    df['IsRestuans'] = ((df['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                               (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
    df['IsOther'] = (df['Species']!='CULEX PIPIENS') * (df['Species']!='CULEX PIPIENS/RESTUANS') * (df['Species']!='CULEX RESTUANS')

if __name__ == '__main__':
    pass