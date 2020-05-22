import os
import pandas as pd
import numpy as np
import itertools
import joblib

from tqdm import tqdm
tqdm.pandas()
pd.set_option('display.max_columns', None)

print('load data')
dsdir = 'dataset/coupon-purchase-prediction'

cd_train = pd.read_csv(os.path.join(dsdir,'coupon_detail_train.csv'))
cl_test = pd.read_csv(os.path.join(dsdir,'coupon_list_test.csv'), usecols=['CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE',
       'DISCOUNT_PRICE', 'DISPFROM', 'DISPPERIOD', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE',
       'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
       'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
       'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name',
       'small_area_name', 'COUPON_ID_hash'])
cl_train = pd.read_csv(os.path.join(dsdir,'coupon_list_train.csv'), usecols=['CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE',
       'DISCOUNT_PRICE', 'DISPFROM', 'DISPPERIOD', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE',
       'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
       'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
       'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name',
       'small_area_name', 'COUPON_ID_hash'])
ca_test = pd.read_csv(os.path.join(dsdir,'coupon_area_test.csv'))
ca_train = pd.read_csv(os.path.join(dsdir,'coupon_area_train.csv'))
cv_train = pd.read_csv(os.path.join(dsdir,'coupon_visit_train.csv'), usecols=['PURCHASE_FLG', 'I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash'])
pref_loc = pd.read_csv(os.path.join(dsdir,'prefecture_locations.csv'), usecols=['PREF_NAME', 'LATITUDE', 'LONGITUDE'])
sample_sub = pd.read_csv(os.path.join(dsdir,'sample_submission.csv'))

cv_train.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)

cl_train.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)
cl_test.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)

cl_train['VALIDPERIOD'].fillna(365, inplace=True)
cl_test['VALIDPERIOD'].fillna(365, inplace=True)

cl_train[['USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY']] = cl_train[['USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY']].fillna(-1)
cl_test[['USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY']] = cl_test[['USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY']].fillna(-1)

cv_train.set_index('I_DATE',inplace=True)
cv_train.sort_index(inplace=True)

cl_dtypes = {'PRICE_RATE': np.int8, 
'CATALOG_PRICE': np.uint32, 
'DISCOUNT_PRICE': np.uint32, 
'DISPPERIOD': np.uint32, 
'VALIDPERIOD': np.uint32, 
'USABLE_DATE_MON': np.int8, 
'USABLE_DATE_TUE': np.int8, 
'USABLE_DATE_WED': np.int8, 
'USABLE_DATE_THU': np.int8, 
'USABLE_DATE_FRI': np.int8, 
'USABLE_DATE_SAT': np.int8, 
'USABLE_DATE_SUN': np.int8, 
'USABLE_DATE_HOLIDAY': np.int8, 
'USABLE_DATE_BEFORE_HOLIDAY': np.int8,
'CAPSULE_TEXT': 'category',
'GENRE_NAME': 'category',
'LARGE_AREA_NAME': 'category',
'SMALL_AREA_NAME': 'category',
'PREF_NAME': 'category',}

ca_dtypes ={'SMALL_AREA_NAME': 'category',
'PREF_NAME': 'category',}

cv_dtypes ={'PURCHASE_FLG': np.int8,}

ul_dtypes ={'AGE': np.uint8,
'SEX_ID': 'category',
'PREF_NAME': 'category',}

cd_dtypes = {'ITEM_COUNT': np.uint32,
'SMALL_AREA_NAME': 'category',}

pl_dtypes = {'LATITUDE': np.float16,
'LONGITUDE': np.float16,
'PREF_NAME': 'category',}

for key, item in cl_dtypes.items():
    cl_train[key] = cl_train[key].astype(item)
    cl_test[key] = cl_test[key].astype(item)

for key, item in ca_dtypes.items():
    ca_train[key] = ca_train[key].astype(item)
    ca_test[key] = ca_test[key].astype(item)

for key, item in cv_dtypes.items():
    cv_train[key] = cv_train[key].astype(item)
    
for key, item in cd_dtypes.items():
    cd_train[key] = cd_train[key].astype(item)
    
for key, item in pl_dtypes.items():
    pref_loc[key] = pref_loc[key].astype(item)

print('preproc data')

cv_train_UCGroup = cv_train.groupby(['USER_ID_hash','COUPON_ID_hash'])

cv_train['PURCHASE_CUMSUM'] = cv_train_UCGroup.PURCHASE_FLG.cumsum().astype(np.int32)
cv_train['VISIT_CUMCOUNT'] = cv_train_UCGroup.cumcount().astype(np.int32)

cv_train['PURCHASE_CUMSUM'] = cv_train['PURCHASE_CUMSUM'] -1
cv_train['VISIT_CUMCOUNT'] = cv_train['VISIT_CUMCOUNT'] -1

cv_train['PURCHASE_CUMSUM'] = cv_train['PURCHASE_CUMSUM'].clip(lower=0)
cv_train['VISIT_CUMCOUNT'] = cv_train['VISIT_CUMCOUNT'].clip(lower=0)

cv_train['PURCHASE_CUMSUM'] = cv_train['PURCHASE_CUMSUM'].astype(np.uint32)
cv_train['VISIT_CUMCOUNT'] = cv_train['VISIT_CUMCOUNT'].astype(np.uint32)

cl_train = pd.merge(cl_train.drop(['PREF_NAME','SMALL_AREA_NAME'],axis=1), ca_train, on='COUPON_ID_hash', how='right')
cl_test = pd.merge(cl_test.drop(['PREF_NAME','SMALL_AREA_NAME'],axis=1), ca_test, on='COUPON_ID_hash', how='right')

cl_train = pd.merge(cl_train,pref_loc,how='left')
cl_test = pd.merge(cl_test,pref_loc,how='left')

cl_train['TEST'] = np.int8(0)
cl_test['TEST'] = np.int8(1)

cl_mix = pd.concat([cl_train, cl_test])

cl_mix.CAPSULE_TEXT = cl_mix.CAPSULE_TEXT.astype('category')
cl_mix.GENRE_NAME = cl_mix.GENRE_NAME.astype('category')

PNandSAN = cl_mix[['COUPON_ID_hash','SMALL_AREA_NAME','PREF_NAME']].copy()
PNandSAN = pd.get_dummies(PNandSAN,columns=['SMALL_AREA_NAME','PREF_NAME'])
PNandSAN = PNandSAN.groupby('COUPON_ID_hash').sum()

cl_mix = pd.merge(cl_mix, PNandSAN, how='left', on='COUPON_ID_hash')
cl_mix = pd.get_dummies(cl_mix, columns=['CAPSULE_TEXT', 'GENRE_NAME', 'LARGE_AREA_NAME'])

user_list = pd.read_csv('world/user_list.csv', usecols=['SEX_ID', 'AGE', 'PREF_NAME', 'USER_ID_hash', 'LATITUDE', 'LONGITUDE'])
user_list.SEX_ID = user_list.SEX_ID.transform(lambda x: int(x=='f'))

for key, item in ul_dtypes.items():
    user_list[key] =user_list[key].astype(item)

print('create transaction')

transaction = pd.merge(cl_mix,cv_train,on='COUPON_ID_hash', how='left')

transaction.PURCHASE_FLG.fillna(np.int8(0),inplace=True)
transaction.PURCHASE_CUMSUM.fillna(np.uint32(0),inplace=True)
transaction.VISIT_CUMCOUNT.fillna(np.uint32(0),inplace=True)

transaction.PURCHASE_FLG = transaction.PURCHASE_FLG.astype(np.int8)
transaction.PURCHASE_CUMSUM = transaction.PURCHASE_CUMSUM.astype(np.uint32)
transaction.VISIT_CUMCOUNT = transaction.VISIT_CUMCOUNT.astype(np.uint32)

transaction = transaction[~transaction.USER_ID_hash.isnull()]

transaction = pd.merge(transaction, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))

transaction['DISTANCE'] = (transaction.LATITUDE_USER-transaction.LATITUDE_COUPON)**2 + (transaction.LONGITUDE_USER-transaction.LONGITUDE_COUPON)**2

transaction.drop(['LATITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_USER', 'LONGITUDE_COUPON'], axis=1, inplace=True)

transaction_distance_threshold = 3.33

transaction = transaction[(transaction.PURCHASE_FLG == 1) | ((transaction.DISTANCE < transaction_distance_threshold) & (transaction.PURCHASE_FLG == 0))]

print('Create Test Data')
#Permutation of User-CouponTest
clist = cl_test.COUPON_ID_hash.unique().tolist()
ulist = user_list.USER_ID_hash.unique().tolist()

relations = [r for r in itertools.product(clist, ulist)]
relations = pd.DataFrame(relations,columns=['COUPON_ID_hash','USER_ID_hash'])

transaction_relations_inner = pd.merge(transaction[['COUPON_ID_hash', 'USER_ID_hash']], relations, on=['COUPON_ID_hash', 'USER_ID_hash'], how="right", indicator=True)

relations = transaction_relations_inner[transaction_relations_inner._merge=='right_only'].drop(["_merge"], axis=1)

cold_test = pd.merge(relations, cl_test,how='left')

cold_test.CAPSULE_TEXT = cold_test.CAPSULE_TEXT.astype('category')
cold_test.GENRE_NAME = cold_test.GENRE_NAME.astype('category')

PNandSAN = cold_test[['COUPON_ID_hash','SMALL_AREA_NAME','PREF_NAME']].copy()
PNandSAN = pd.get_dummies(PNandSAN,columns=['SMALL_AREA_NAME','PREF_NAME'])
PNandSAN = PNandSAN.groupby('COUPON_ID_hash').sum()
PNandSAN = PNandSAN.astype(np.uint8)

cold_test = pd.merge(cold_test, PNandSAN, how='left', on='COUPON_ID_hash')
cold_test = pd.get_dummies(cold_test, columns=['CAPSULE_TEXT', 'GENRE_NAME', 'LARGE_AREA_NAME'])

cold_test = pd.merge(cold_test, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))

cold_test['PURCHASE_FLG'] = np.int8(0)
cold_test['PURCHASE_CUMSUM'] = np.uint32(0)
cold_test['VISIT_CUMCOUNT'] = np.uint32(0)
cold_test['DISTANCE'] = (cold_test.LATITUDE_USER-cold_test.LATITUDE_COUPON)**2 + (cold_test.LONGITUDE_USER-cold_test.LONGITUDE_COUPON)**2

cold_test.drop(['LATITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_USER', 'LONGITUDE_COUPON'], axis=1, inplace=True)

cold_test_distance_threshold = 3.33

cold_test = cold_test[cold_test.DISTANCE < cold_test_distance_threshold]

print('combine transaction and cold_test')

transaction = pd.concat([transaction, cold_test])

transaction['SAME_PREF'] = transaction.PREF_NAME_COUPON == transaction.PREF_NAME_USER
transaction.drop(['PREF_NAME_COUPON', 'PREF_NAME_USER', 'SMALL_AREA_NAME'], axis=1, inplace=True)

new_columns_found = [x for x in transaction.columns.tolist() if x not in cold_test.columns.tolist()]

transaction[new_columns_found] = transaction[new_columns_found].fillna(0)
transaction[new_columns_found] = transaction[new_columns_found].astype(np.uint8)

train = transaction[transaction.TEST==0]
test =  transaction[transaction.TEST==1]

print('save train')
joblib.dump(train, 'world/train.pkl')
print('save test')
joblib.dump(test, 'world/test.pkl')