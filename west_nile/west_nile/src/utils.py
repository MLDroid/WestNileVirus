def process_date(train, test):
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

    for ds in [train, test]:
        ds['year'] = ds.Date.apply(create_year)
        ds['month'] = ds.Date.apply(create_month)
        ds['day'] = ds.Date.apply(create_day)

def convert_species(train, test):
    '''
    Convert the Species field into 4 attributes: IsPipiens, IsPipiensRestuans (gets 0.5 for Pipiens and for Restuans),
    IsRestuans, and IsOther (for all other species).
    '''
    for df in [train, test]:
        # % Wnv  / total count in train data
#         df['IsErraticus'] = (df['Species']=='CULEX ERRATICUS')*1            #   0%   /    1
        df['IsPipiens'] = ((df['Species']=='CULEX PIPIENS'  )*1 +           # 8.9%   / 2699
                                   (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
        df['IsPipiensRestuans'] = ((df['Species']=='CULEX PIPIENS/RESTUANS')*1 +    # 5.5%   / 4752
                                   (df['Species']=='CULEX PIPIENS'  )*0 + (df['Species']=='CULEX RESTUANS'  )*0)
        df['IsRestuans'] = ((df['Species']=='CULEX RESTUANS'  )*1 +          # 1.8%   / 2740
                                   (df['Species']=='CULEX PIPIENS/RESTUANS')*0.5)
#         df['IsSalinarius'] = (df['Species']=='CULEX SALINARIUS')*1           #   0%   /   86
#         df['IsTarsalis'] = (df['Species']=='CULEX TARSALIS'  )*1           #   0%   /    6
#         df['IsTerritans'] = (df['Species']=='CULEX TERRITANS' )*1           #   0%   /  222
        df['IsOther'] = (df['Species']!='CULEX PIPIENS')*\
                        (df['Species']!='CULEX PIPIENS/RESTUANS')*\
                        (df['Species']!='CULEX RESTUANS')

if __name__ == '__main__':
    pass