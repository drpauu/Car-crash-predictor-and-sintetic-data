import pandas as pd
import numpy as np 
from datetime import datetime, date
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt, log
import sys
sys.path.append('/home/user/Github/__pycache__')
from crash_time import add_crash_timestamp
from time import mktime
import gc

def data_preprocessing():
    flag = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    user = list(flag['vin'])
    number = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']
    result = pd.DataFrame()
    for num in tqdm(number):
        filename = 'G:/shanghaimotor/dataset/loc/part-000'
        filename += num
        dataset = pd.read_csv(filename, names=['vin', 'latitude', 'longitude', 'speed','heading_val',
                                               'ignition_on_flag', 'record_time', 'i_flag', 'ignition_on_flag_new'])
        dataset = dataset.loc[:,('vin', 'latitude', 'longitude', 'speed', 'record_time')]
        dataset = dataset.astype(np.str)
        dataset['vin'] = dataset['vin'].apply(lambda x:((str(x)).split("=")))
        dataset['vin'] = dataset['vin'].apply(lambda x:x[1])
        dataset['vin'] = dataset['vin'].astype('int32')
        dataset = dataset[dataset['vin'].isin(user)]
        dataset['nan'] = dataset['latitude'].apply(lambda x:str(x)[-4:])
        dataset['none'] = dataset['nan'].apply(lambda x:1 if(x=='None') else 0)
        dataset = dataset[dataset['none'].isin([0])]
        dataset.drop(['nan', 'none'], axis=1, inplace=True)
        dataset['latitude'] = dataset['latitude'].apply(lambda x:(str(x)).split("'"))
        dataset['latitude'] = dataset['latitude'].apply(lambda x:x[1])
        dataset['longitude'] = dataset['longitude'].apply(lambda x:(str(x)).split("'"))
        dataset['longitude'] = dataset['longitude'].apply(lambda x:x[1])
        dataset['speed1'] = dataset['speed'].apply(lambda x:(str(x)).replace("None", "u'None'"))
        dataset['speed1'] = dataset['speed1'].apply(lambda x:(str(x)).split("'"))
        dataset['speed'] = dataset['speed1'].apply(lambda x:x[1])
        dataset['record_time'] = dataset['record_time'].apply(lambda x:(str(x)).split("'"))
        dataset['record_time'] = dataset['record_time'].apply(lambda x:x[1])
        dataset.drop('speed1', axis=1, inplace=True)
        dataset = dataset[dataset['speed'] != 'None']

        dataset['latitude'] = dataset['latitude'].astype('float32')
        dataset['longitude'] = dataset['longitude'].astype('float32')
        dataset['speed'] = dataset['speed'].astype('float16')
        dataset = dataset[dataset['latitude'] < 500]
        dataset = dataset[dataset['latitude'] > 10]
        dataset = dataset[dataset['longitude'] < 500]
        dataset['record_time'] = dataset['record_time'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M"))
        result = result.append(dataset, ignore_index=True)
    return(result)
    
def to_csv(dataset):
    l = len(dataset) // 10000000
    k1 = 0 
    k2 = 10000000
    for i in range(l):
        result = dataset.loc[k1:k2]
        result.to_csv("G:/shanghaimotor/code/track_21/part"+str(i)+".csv", index=False)
        add_timestamp(result)
        k1 += 10000000
        k2 += 10000000
        gc.collect()
    result = dataset.loc[k1:]
    add_timestamp(result)
    result.to_csv("G:/shanghaimotor/code/track_21/part"+str(l)+".csv", index=False)

    
def read_dataset():
    result = pd.DataFrame()
    for i in tqdm(range(25)):
        filename = "G:/shanghaimotor/code/track_21/part" + str(i) + ".csv"
        dataset = pd.read_csv(filename)
        result = result.append(dataset, ignore_index=True)
    result = result[result['latitude'] < 54]
    result = result[result['latitude'] > 10]
    result = result[result['longitude'] < 136]
    result = result[result['longitude'] > 73]
    return(result)    

def flag2_2018(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    flag1 = flag1[flag1['year'] == 2018]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result = pd.merge(result, flag1, on='vin')
    result = result[result['month'] < result['crash_month']]
    result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)
    
def flag2_2019(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    flag1 = flag1[flag1['year'] == 2019]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)

def flag1_2018(dataset):
    result = dataset.copy()
    flag1 = pd.read_csv(r"G:\shanghaimotor\code\flag1.csv")
    flag1 = flag1[flag1['year'] == 2018]
    flag1 = flag1[['vin', 'year', 'month']]
    flag1.columns = ['vin', 'crash_year', 'crash_month']
    result = result[result['vin'].isin(list(flag1['vin'].unique()))]    
    result = result[result['year'] == 2018]
    result = pd.merge(result, flag1, on='vin')
    result = result[result['month'] < result['crash_month']]
    result.drop(['crash_year', 'crash_month'], axis=1, inplace=True)
    result.sort_values(['vin', 'record_time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return(result)
    
def add_timestamp(dataset):
    dataset['record_time'] = dataset['record_time'].astype(str)
    dataset['record_time'] = dataset['record_time'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    dataset['year'] = dataset['record_time'].apply(lambda x:x.year)
    dataset['month'] = dataset['record_time'].apply(lambda x:x.month)
    dataset['day'] = dataset['record_time'].apply(lambda x:x.day)
    dataset['hour'] = dataset['record_time'].apply(lambda x:x.hour)

def driving_time(dataset):
    num_of_point = dataset.copy()
    add_timestamp(num_of_point)
    num_of_point = num_of_point.loc[:,('vin', 'latitude', 'longitude', 'year', 'month', 'day')]
    num_of_point = num_of_point.groupby(['vin', 'year', 'month', 'day']).count()
    num_of_point.reset_index(inplace=True)
    num_of_point = num_of_point.loc[:,('vin', 'latitude')]
    num_of_point = num_of_point.groupby('vin').mean()
    num_of_point.reset_index(inplace=True)
    num_of_point.columns = ['vin', 'driving_time']
    return(num_of_point)

def covering(dataset):
    covering = dataset.copy()
    add_timestamp(covering)
    covering['lat'] = covering['latitude'].apply(lambda x:round(x,2))
    covering['lgt'] = covering['longitude'].apply(lambda x:round(x,2))
    covering = covering.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    covering = covering.loc[:,('vin', 'lat', 'lgt', 'year', 'month', 'day')]
    covering = covering.groupby(['vin', 'year', 'month', 'day']).count()
    covering.reset_index(inplace=True)
    covering = covering.loc[:,('vin', 'lat')]
    covering = covering.groupby('vin').mean()
    covering.reset_index(inplace=True)
    covering.columns = ['vin', 'covering']
    return(covering)

def visiting_frequency(dataset):
    place = dataset.copy()
    add_timestamp(place)
    place['lat3'] = place['latitude'].apply(lambda x:round(x,3))
    place['lgt3'] = place['longitude'].apply(lambda x:round(x,3))
    place.drop(['latitude', 'longitude', 'speed'], axis=1, inplace=True)
    place = place.drop_duplicates(['vin', 'lat3', 'lgt3', 'year', 'month', 'day', 'hour'])
    place.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    place = place.groupby(['vin', 'lat3', 'lgt3']).count()
    place.columns = ['visiting_count']
    place.reset_index(inplace=True)
    user_place = pd.DataFrame(columns = ['vin', 'place_dif'])
    i = 0
    for vin in tqdm(place['vin'].unique()):
        dt_user = place[place['vin'] == vin]
        dt_user.sort_values('visiting_count', inplace=True, ascending=False)
        dt_user.reset_index(drop=True, inplace=True)
        dt_user['visiting_pro'] = dt_user['visiting_count'] / dt_user['visiting_count'].sum()
        user_place.loc[i, 'vin'] = vin
        if len(dt_user) == 1:
            user_place.loc[i, 'place_dif'] = dt_user.loc[0, 'visiting_pro']
        else:
            user_place.loc[i, 'place_dif'] = dt_user.loc[0, 'visiting_pro'] - dt_user.loc[1, 'visiting_pro']
        i += 1
    user_place['place_dif'] = user_place['place_dif'].apply(lambda x:float('%.4f' %x))
    return(user_place)

def haversine(lon1, lon2, lat1, lat2):

    lon1, lon2, lat1, lat2 = map(radians, [lon1, lon2, lat1, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return(c * r)

     
def turning_radius(dataset):
    turning = dataset.copy()
    add_timestamp(turning)
    turning['lat3'] = turning['latitude'].apply(lambda x:round(x,3))
    turning['lgt3'] = turning['longitude'].apply(lambda x:round(x,3))
    turning.drop(['latitude', 'longitude'], axis=1, inplace=True)
    turning = turning.drop_duplicates(['vin', 'lat3', 'lgt3', 'year', 'month', 'day', 'hour'])
    turning.drop(['year', 'month', 'day', 'hour', 'speed'], axis=1, inplace=True)
    turning = turning.groupby(['vin', 'lat3', 'lgt3']).count()
    turning.columns = ['visiting_count']
    turning.reset_index(inplace=True)            
    home = pd.DataFrame(columns = ['vin', 'lat3_home', 'lgt3_home'])
    i = 0
    for vin in tqdm(turning['vin'].unique()):
        dt_user = turning[turning['vin'] == vin]
        dt_user.sort_values('visiting_count', ascending=False, inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        home.loc[i, 'vin'] = vin
        home.loc[i, 'lat3_home'] = dt_user.loc[0, 'lat3']
        home.loc[i, 'lgt3_home'] = dt_user.loc[0, 'lgt3']
        i += 1
    i =0 
    turning_radius = pd.DataFrame(columns=['vin', 'turning_radius'])
    for vin in tqdm(turning['vin'].unique()):
        dt_user = turning[turning['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        dt_user = pd.merge(dt_user, home, on='vin')
        l = len(dt_user)
        d_s = 0
        for j in range(l):
            d = haversine(dt_user.loc[j, 'lgt3_home'], dt_user.loc[j, 'lgt3'], dt_user.loc[j, 'lat3_home'], dt_user.loc[j, 'lat3'])
            d_s += d
        d_s /= l
        turning_radius.loc[i, 'vin'] = vin
        turning_radius.loc[i, 'turning_radius'] = d_s
        i += 1
    return(turning_radius)

def weekdays_weekends(dataset):
    ww = dataset.copy()
    add_timestamp(ww)
    ww['weekday'] = ww['record_time'].apply(lambda x:date.isoweekday(x))
    ww['isweekend'] = ww['weekday'].apply(lambda x:1 if((x==6)|(x==7)) else 0)
    ww.drop('weekday', axis=1, inplace=True)
    weekdays = ww[ww['isweekend'] == 0]
    weekends = ww[ww['isweekend'] == 1]
    weekdays = weekdays.groupby(['vin', 'year', 'month', 'day']).count()
    weekdays.reset_index(inplace=True)
    weekdays = weekdays.loc[:,('vin', 'record_time')]
    weekends = weekends.groupby(['vin', 'year', 'month', 'day']).count()
    weekends.reset_index(inplace=True)
    weekends = weekends.loc[:,('vin', 'record_time')]    
    weekdays = weekdays.groupby('vin').mean()
    weekends = weekends.groupby('vin').mean()
    weekdays.reset_index(inplace=True)
    weekends.reset_index(inplace=True)
    weekdays.columns = ['vin', 'weekdays_count']
    weekends.columns = ['vin', 'weekends_count']
    ww = pd.merge(weekdays, weekends, on='vin', how='outer')
    ww = ww.fillna(0)
    ww['weekdays_weekends'] = ww['weekdays_count'] - ww['weekends_count']
    ww = ww.loc[:,('vin', 'weekdays_weekends')]
    return(ww)

def life_entropy(dataset):
    life_en = dataset.copy()
    add_timestamp(life_en)
    life_en['lat'] = life_en['latitude'].apply(lambda x:float('%.2f' %x))
    life_en['lgt'] = life_en['longitude'].apply(lambda x:float('%.2f' %x))
    life_en = life_en.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    life_en = life_en.loc[:,('vin', 'lat', 'lgt', 'year', 'month', 'day')]
    life_en = life_en.groupby(['vin', 'year', 'month', 'day']).count()
    life_en.reset_index(inplace=True)
    life_en = life_en.loc[:,('vin', 'month', 'lat')]    
    entropy = pd.DataFrame(columns = ['vin', 'life_entropy'])
    i = 0
    for vin in tqdm(life_en['vin'].unique()):
        dt_user = life_en[life_en['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        month_count = dt_user.groupby(['vin', 'month']).sum()
        month_count.reset_index(inplace=True)
        month_count.columns = ['vin', 'month', 'month_count']
        dt_user = pd.merge(dt_user, month_count, on=['vin', 'month'])
        dt_user['pro'] = dt_user['lat'] / dt_user['month_count']
        dt_user['entropy'] = dt_user['pro'].apply(lambda x:(-x*log(x)))
        dt_user = dt_user.groupby(['vin', 'month']).sum()
        dt_user.reset_index(inplace=True)
        dt_user = dt_user.groupby('vin').mean()
        dt_user.reset_index(inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        entropy.loc[i, 'vin'] = vin
        entropy.loc[i, 'life_entropy'] = dt_user.loc[0, 'entropy']
        i += 1
    return(entropy)

def driving_distance(dataset):
    dis = dataset.copy()
    add_timestamp(dis)
    distance = pd.DataFrame(columns=['vin', 'distance', 'day'])
    i = 0
    for vin in tqdm(dis['vin'].unique()):
        dt_user = dis[dis['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        d = 0.0
        if len(dt_user) > 1:
            for j in range(len(dt_user)-1):
                time = (dt_user.loc[j+1, 'record_time'] - dt_user.loc[j, 'record_time']).seconds
                if time <= 600:
                    lon1 = dt_user.loc[j, 'longitude']
                    lon2 = dt_user.loc[j+1, 'longitude']
                    lat1 = dt_user.loc[j, 'latitude']
                    lat2 = dt_user.loc[j+1, 'latitude']
                    d += haversine(lon1, lon2, lat1, lat2)
                else:
                    continue
            distance.loc[i, 'vin'] = vin
            distance.loc[i, 'distance'] = d
            day_count = dt_user.drop_duplicates(['year', 'month', 'day'])
            distance.loc[i, 'day'] = len(day_count)
        else:
            distance.loc[i, 'vin'] = vin
            distance.loc[i, 'distance'] = 0
            distance.loc[i, 'day'] = 1 
        i += 1
    distance['dis_per_day'] = distance['distance'] / distance['day']
    return(distance)

def driving_trip(dataset):
    t = dataset.copy()
    t['record_time'] = t['record_time'].apply(lambda x:str(x))
    t['record_time'] = t['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    trip = pd.DataFrame(columns = ['vin', 'trip'])
    i = 0
    for vin in tqdm(t['vin'].unique()):
        dt_user = t[t['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        count = 1
        for j in range(len(dt_user)-1):
            time = (dt_user.loc[j+1, 'record_time'] - dt_user.loc[j, 'record_time']).seconds
            if time > 1800:
                count += 1
            else:
                continue
        trip.loc[i, 'vin'] = vin
        trip.loc[i, 'trip'] = count
        i += 1
    return(trip)
    
def dayofweek_pro(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]

    loc = loc.drop_duplicates(['vin', 'year', 'month', 'day'])
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    loc['dayofweek'] = loc['record_time'].apply(lambda x:date.isoweekday(x))
    dayofweek_day = pd.DataFrame(columns=['vin', 'weekdays', 'weekends', 'weekdays_pro', 'weekends_pro'])
    j = 0
    for vin in tqdm(loc['vin'].unique()):
        wd = 0
        wk = 0
        dt_user = loc[loc['vin'] == vin]
        dt_user.reset_index(drop=True, inplace=True)
        for i in range(len(dt_user)):
            day = dt_user.loc[i, 'dayofweek']
            if (day == 6) | (day == 7):
                wk += 1
            else:
                wd += 1
        dayofweek_day.loc[j, 'vin'] = vin
        dayofweek_day.loc[j, 'weekdays'] = wd
        dayofweek_day.loc[j, 'weekends'] = wk
        dayofweek_day.loc[j, 'weekdays_pro'] = (wd / (wd + wk))
        dayofweek_day.loc[j, 'weekends_pro'] = (wk / (wd + wk))  
        j += 1
    return(dayofweek_day)
           

def most_visiting(loc):
    visiting = loc.copy()
    add_timestamp(visiting)
    month = pd.DataFrame(columns=['vin', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    i = 0
    for vin in tqdm(visiting['vin'].unique()):
        dt_user = visiting[visiting['vin']==vin]
        dt_user['lat'] = dt_user['latitude'].apply(lambda x:round(x, 2))
        dt_user['lgt'] = dt_user['longitude'].apply(lambda x:round(x, 2))
        dt_user.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'], inplace=True)
        dt_user.reset_index(drop=True, inplace=True)
        dt_user.drop(['latitude', 'longitude', 'speed', 'record_time', 'year', 'day', 'hour', 'speed'], axis=1, inplace=True)
        dt_user = dt_user.groupby(['month', 'lat', 'lgt']).count()
        dt_user.reset_index(inplace=True)
        for m in dt_user['month'].unique():
            dt_user_month = dt_user[dt_user['month']==m]
            dt_user_month.sort_values(['vin'], ascending=False, inplace=True)
            dt_user_month.reset_index(drop=True, inplace=True)
            if (len(dt_user_month)) <= 10:
                month.loc[i, str(m)] = 1
            else:
                top = dt_user_month.loc[:9]
                p = top['vin'].sum() / dt_user_month['vin'].sum()
                month.loc[i, str(m)] = p
        month.loc[i, 'vin'] = vin
        i += 1
    month = month.astype(float)
    for i in range(len(month)):
        t1 = month.iloc[i,1:12]
        t2 = t1.mean()
        month.loc[i, 'mean'] = t2
    return(month)

def weekends_travelling(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]
    add_timestamp(loc)
    flag = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    loc = loc[loc['vin'].isin(list(flag['vin']))]    
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    loc['dayofweek'] = loc['record_time'].apply(lambda x:date.isoweekday(x))
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    weekends_loc = loc[loc['dayofweek'].isin([6, 7])]
    weekends_visiting = pd.DataFrame(columns = ['vin', 'weekends_visiting'])
    i = 0
    for vin in tqdm(list(flag['vin'])):
        dt_user_weekends = weekends_loc[weekends_loc['vin'] == vin]
        dt_user = weekends_loc[weekends_loc['vin'] == vin]
        if len(dt_user_weekends) == 0:
            weekends_visiting.loc[i, 'weekends_visiting'] = 1
        else:
            dt_user = dt_user.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
            month_weekends = []
            for m in dt_user['month'].unique():
                dt_user_weekends_month = dt_user_weekends[dt_user_weekends['month'] == m]
                dt_user_weekends_month = dt_user_weekends_month[['vin', 'year', 'month', 'day', 'lat', 'lgt']]
                dt_user_weekends_month = dt_user_weekends_month.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
                dt_user_weekends_month = dt_user_weekends_month[['lat', 'lgt']]
                dt_user_month = dt_user[dt_user['month'] == m]
                dt_user_month = dt_user_month.groupby(['lat', 'lgt']).count()
                month_visiting = dt_user_month.reset_index()
                month_visiting.sort_values('vin', ascending=False, inplace=True)
                if len(month_visiting) <= 10:
                    month_visiting = month_visiting.iloc[:,:3]
                else:
                    month_visiting = month_visiting.iloc[:10,:3]
                
                month_visiting1 = pd.merge(month_visiting, dt_user_weekends_month, on=['lat', 'lgt'], how='outer')
                month_visiting1 = month_visiting1.fillna(0)
                month_visiting1['tag'] = month_visiting1['vin'].apply(lambda x:1 if(x==0) else 0)
                month_weekends.append(np.mean(month_visiting1['tag']))
            weekends_visiting.loc[i, 'vin'] = vin
            weekends_visiting.loc[i, 'weekends_visiting'] = np.mean(month_weekends)
        i += 1
    return(weekends_visiting)

def fatigue_driving_count(dataset):
    loc = dataset.copy()
    loc = loc[loc['year'] == 2018]
    loc['record_time'] = loc['record_time'].apply(lambda x:str(x))
    loc['record_time'] = loc['record_time'].apply(lambda x:datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))    
    k = 0
    fatigue_driving = pd.DataFrame(columns=['vin', 'fat_driving_count'])
    for vin in tqdm(loc['vin'].unique()):
        dt_user = loc[loc['vin'] ==  vin]
        dt_user.reset_index(drop=True, inplace=True)
        dt_user['seconds'] = dt_user['record_time'].apply(lambda x:mktime(x.timetuple()))
        i = 0
        count = 0
        while(i < len(dt_user)-1):
            start_time = dt_user.loc[i, 'seconds']
            for j in range(i, len(dt_user)-1):
                if (dt_user.loc[j+1, 'seconds'] - dt_user.loc[j, 'seconds']) < 1800.0:
                    continue
                else:
                    end_time = dt_user.loc[j, 'seconds']
                    if (end_time - start_time) >= 14400.0:
                        count += 1
                    break
            i = j+1
        fatigue_driving.loc[k, 'vin'] = vin
        fatigue_driving.loc[k, 'fat_driving_count'] = count
        k += 1
    return(fatigue_driving)
                        

def accident_rate(dataset):
    loc = dataset.copy()
    crash = pd.read_csv(r"G:\shanghaimotor\code\flag2.csv")
    add_crash_timestamp(crash)
    crash_place = pd.merge(loc, crash, on=['vin', 'year', 'month', 'day', 'hour'])
    crash_place['lat'] = crash_place['latitude'].apply(lambda x:round(x, 2))
    crash_place['lgt'] = crash_place['longitude'].apply(lambda x:round(x, 2))
    crash_place['time'] = (crash_place['crash_time'] - crash_place['record_time'])
    crash_place['time'] = crash_place['time'].apply(lambda x:int(x.seconds))
    crash_place = crash_place[crash_place['time'] < 600]
    crash_place.drop_duplicates('vin', keep='last', inplace=True)
    crash_place = crash_place[['vin', 'lat', 'lgt']]
    accident_account = crash_place.groupby(['lat', 'lgt']).count()
    accident_account.reset_index(inplace=True)
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    loc_account = loc.drop_duplicates(['vin', 'year', 'month', 'day', 'lat', 'lgt'])
    loc_account = loc_account[['vin', 'lat', 'lgt']]
    loc_account = loc_account.groupby(['lat', 'lgt']).count()
    loc_account.reset_index(inplace=True)
    accident_account.columns = ['lat', 'lgt', 'accident_account']
    loc_account.columns = ['lat', 'lgt', 'loc_account']
    accident_rate = pd.merge(accident_account, loc_account, on=['lat', 'lgt'])
    accident_rate['accident_rate'] = accident_rate['accident_account'] / accident_rate['loc_account']
    accident_rate = accident_rate[accident_rate['lat'] > 30.40]
    accident_rate = accident_rate[accident_rate['lat'] < 31.53]
    accident_rate = accident_rate[accident_rate['lgt'] < 122.12]
    accident_rate = accident_rate[accident_rate['lgt'] > 121]
    accident_rate = accident_rate[['lat', 'lgt', 'accident_rate']]
    loc_account = loc[~loc['speed'].isin([0])]
    loc_account = loc_account.drop_duplicates(['vin', 'year', 'month', 'day', 'hour', 'lat', 'lgt'])
    loc_account = loc_account[['vin', 'lat', 'lgt']]
    loc_account = pd.merge(loc_account, accident_rate, on=['lat', 'lgt'])
    loc_account = loc_account[['vin', 'accident_rate']]
    loc_account = loc_account.groupby('vin').sum()
    loc_account.reset_index(inplace=True)
    return(loc_account)


def speed_limit(dataset):
    speed = dataset.copy()
    speed['lat'] = speed['latitude'].apply(lambda x:round(x, 2))
    speed['lgt'] = speed['longitude'].apply(lambda x:round(x, 2))
    speed = speed[['year', 'month', 'lat', 'lgt', 'speed']]
    speed.dropna(how='any', inplace=True)
    speed = speed[~speed['speed'].isin([0])]
    speed = speed.groupby(['year', 'month', 'lat', 'lgt']).mean()
    speed.reset_index(inplace=True)
    speed = speed[speed['lat'] > 30.40]
    speed = speed[speed['lat'] < 31.53]
    speed = speed[speed['lgt'] < 122.12]
    speed = speed[speed['lgt'] > 121]
    speed.columns = ['year', 'month', 'lat', 'lgt', 'speed_limit']
    return(speed)        

def overspeed(dataset, speed_limit):
    loc = dataset.copy()
    loc['lat'] = loc['latitude'].apply(lambda x:round(x, 2))
    loc['lgt'] = loc['longitude'].apply(lambda x:round(x, 2))
    loc_account = loc[~loc['speed'].isin([0])]
    loc_account = loc_account[['vin', 'year', 'month', 'lat', 'lgt', 'speed']]
    loc_account = loc_account.groupby(['vin', 'year', 'month', 'lat', 'lgt']).mean()
    loc_account.reset_index(inplace=True)
    speed = pd.merge(loc_account, speed_limit, on=['year', 'month', 'lat', 'lgt'])
    speed['over_speed'] = speed['speed'] - speed['speed_limit']
    speed['over_speed_tag'] = speed['over_speed'].apply(lambda x:0 if(x<0) else 1)
    speed = speed[['vin','over_speed_tag']]
    speed = speed.groupby('vin').mean()
    speed.reset_index(inplace=True)
    return(speed)    

def weather_feature(dataset, wea):
    loc = dataset.copy()
    loc = loc[loc['latitude'] > 30.40]
    loc = loc[loc['latitude'] < 31.53]
    loc = loc[loc['longitude'] < 122.12]
    loc = loc[loc['longitude'] > 120.51]   
    loc = loc[loc['year'] == 2018]
    loc.drop_duplicates(['vin', 'year', 'month', 'day'], inplace=True)
    loc = loc[['vin', 'year', 'month', 'day']]
    loc = pd.merge(loc, wea, on=['year', 'month', 'day'])
    loc = loc.groupby('vin').mean()
    loc.reset_index(inplace=True)
    loc = loc[['vin', 'cond', 'temp']]
    return(loc)
                  
    

if __name__ == '__main__':
    result = read_dataset()
