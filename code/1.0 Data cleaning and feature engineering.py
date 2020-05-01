#
# Program: 1.0 Data cleaning and feature engineering
#
# Purpose: Implement data cleaning and feature engineering
#          to the sales data 
#
# Written by: Qiuhua Liu(11258799) 
#             Yanhan Peng(11125583) 
#             Chaoyang Zheng(11249259) 
# 
# Updated: Dec 2019
#        
# ------------------------------------------------------.


# ### Loading library

import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta


# ##  Data processing:Train.csv

# ##### Read train data 

data = os.path.abspath(r"C:/Users/Zheng Chaoyang/Desktop/ML Group project/Data/Input")
os.chdir(data)
train = pd.read_csv("train.csv")


# ##### Arbitrarily choose '2016-07-01' to '2017-07-02' as inscope

date_mask = (train['date'] >= '2016-07-03') & (train['date'] <= '2017-07-02')
print(train.shape)
train_sample = train[date_mask]
print(train_sample.shape)


# ##### To get some basic information for train_sample dataset

print("Nulls in train_sample columns: {0} => {1}".format(train_sample.columns.values,train_sample.isnull().any().values))
print("="*70)
print(train_sample.info())


# ##### Feature engineering

# eliminate negatives
train_sample.loc[(train_sample.unit_sales<0),'unit_sales'] = 0 

# get day of week(ordinal)
train_sample['date'] = pd.to_datetime(train_sample['date'])
train_sample['day_of_week'] = train_sample['date'].dt.dayofweek 

# get the year and month variable
train_sample['year'] = train_sample['date'].dt.year 
train_sample['month'] = train_sample['date'].dt.month 

train_sample.head()


# 
# ## Data processing: Supplementary data 

# ##### Read supplementary data 

items = pd.read_csv("items.csv")
holiday_events = pd.read_csv("holidays_events.csv")
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv")
transactions = pd.read_csv("transactions.csv",parse_dates=['date'])

print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))
print("="*70)
print("Nulls in holiday_events columns: {0} => {1}".format(holiday_events.columns.values,holiday_events.isnull().any().values))
print("="*70)
print("Nulls in stores columns: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))
print("="*70)
print("Nulls in transactions columns: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))
print("="*70)
print("CONCLUDE:The only missing data occurs in the oil data file, which provides the historical daily price for oil.")


# ##### Arbitrarily merge train_sample with other supplementary data 

# Merging with items
items.rename(columns={"family": "item_family", "class": "item_class","perishable": "item_perishable" }, inplace=True)# rename columns for ease of understandng 
train_sample = train_sample.merge(items, how='left', left_on='item_nbr', right_on='item_nbr')
train_sample.head()

# Merging with holiday_events
holiday_events.rename(columns={"type": "specday_type", "locale": "specday_locale","locale_name": "specday_city","description": "specday_description"}, inplace=True)# rename columns for ease of understandng 
holiday_events['date'] = pd.to_datetime(holiday_events['date'])
train_sample = train_sample.merge(holiday_events, how='left', left_on='date', right_on='date')
train_sample.head()

# Merging with stores
stores.rename(columns={"city": "store_city", "state": "store_state","type": "store_type","cluster": "store_cluster"}, inplace=True)
train_sample = train_sample.merge(stores, how='left', left_on='store_nbr', right_on='store_nbr')
train_sample.head()

# Merging with transactions
train_sample = train_sample.merge(transactions, how='left', left_on=['date','store_nbr'], right_on=['date','store_nbr'])
train_sample.head()


# ## Data processing: merged dataset(train_sample)

# ##### Aggregate to get monthly unit sales per item in each store

train_monthly = train_sample.groupby(["year","month","item_nbr","store_nbr"]).agg({"unit_sales":'sum',
                                                                                  "transactions":"sum",
                                                                                  }).reset_index()


# ##### Merging with other supplementary dataset

# with stores info
train_monthly = train_monthly.merge(stores, how='left', left_on='store_nbr', right_on='store_nbr')

# with items info
train_monthly = train_monthly.merge(items, how='left', left_on='item_nbr', right_on='item_nbr')

train_monthly.head()


# ###### Processing event_holiday data to monthly count

holiday_events['year'] = holiday_events['date'].dt.year 
holiday_events['month'] = holiday_events['date'].dt.month 
# National holiday monthly count
specdays_national = holiday_events.groupby(["year","month","specday_locale"]).agg('count').reset_index()
specdays_national = specdays_national[specdays_national.specday_locale == "National"]
specdays_national = specdays_national.drop(['specday_type',"specday_locale", 'specday_city',"specday_description","transferred"], axis=1)
specdays_national.rename(columns={"date": "count_national_holiday"}, inplace=True)
print("National holiday monthly count")
print(specdays_national.head())

# Regional holiday monthly count
specdays_regional = holiday_events[holiday_events.specday_locale == "Regional"]
specdays_regional = specdays_regional.groupby(["year","month","specday_city"]).agg('count').reset_index()
specdays_regional = specdays_regional.drop(['specday_type', 'specday_locale',"specday_description","transferred"], axis=1)
specdays_regional.rename(columns={"date": "count_regional_holiday","specday_city":"store_state"}, inplace=True)
print("="*70)
print("National holiday monthly count, based on state ")
print(specdays_regional.head())

# local monthly holiday count
specdays_local = holiday_events[holiday_events.specday_locale == "Local"]
specdays_local = specdays_local.groupby(["year","month","specday_city"]).agg('count').reset_index()
specdays_local = specdays_local.drop(['specday_type', 'specday_locale',"specday_description","transferred"], axis=1)
specdays_local.rename(columns={"date": "count_local_holiday","specday_city":"store_city"}, inplace=True)
print("="*70)
print("Local holiday monthly count, based on city ")
print(specdays_local.head())


# ##### Merging holiday date with monthly sales dataset 

# with national holiday
train_monthly = train_monthly.merge(specdays_national, how='left', left_on=["year","month"], right_on=["year","month"])
print("="*70)
print("with national holiday ")
print(train_monthly.head())

# with regional holiday
train_monthly = train_monthly.merge(specdays_regional, how='left', left_on=["year","month","store_state"], right_on=["year","month","store_state"])
print("="*70)
print("with regional holiday ")
print(train_monthly.head())

# with local holiday
train_monthly = train_monthly.merge(specdays_local, how='left', left_on=["year","month","store_city"], right_on=["year","month","store_city"])
print("="*70)
print("with local holiday ")
print(train_monthly.head())

print(train_monthly.info())
train_monthly.head()


# ##### Merging with count of monthly item & store on promotion days


count_onpromotion = train_sample.groupby(["year","month","item_nbr","store_nbr"]).agg({"onpromotion":'count',
                                                                                  }).reset_index()
count_onpromotion.rename(columns={"onpromotion": "count_days_onpromotion"}, inplace=True)

train_monthly = train_monthly.merge(count_onpromotion, how='left', left_on=["year","month","item_nbr","store_nbr"], right_on=["year","month","item_nbr","store_nbr"])

train_monthly_nona = train_monthly.fillna(0)

train_monthly_nona.head()

train_monthly_nona.to_csv(index=False,path_or_buf  = r"C:/Users/Zheng Chaoyang/Desktop/ML Group project/Data/Output/train_monthly_nona3.csv")


# ## Combining dataset

import os
import pandas as pd
import numpy as np
data = os.path.abspath(r"C:/Users/Zheng Chaoyang/Desktop/ML Group project/Data/Output")
os.chdir(data)
train1 = pd.read_csv("train_monthly_nona.csv")
train2 = pd.read_csv("train_monthly_nona2.csv")
train3 = pd.read_csv("train_monthly_nona3.csv")

combined_train_monthly = train1
combined_train_monthly = combined_train_monthly.append(train2, ignore_index=True)
combined_train_monthly = combined_train_monthly.append(train3, ignore_index=True)

combined_train_monthly.to_csv(index=False,path_or_buf  = r"C:/Users/Zheng Chaoyang/Desktop/ML Group project/Data/Output/combined_train_monthly.csv")

