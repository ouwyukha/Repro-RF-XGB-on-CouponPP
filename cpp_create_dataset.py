import pandas as pd
import os
import numpy as np
import itertools
from tqdm import tqdm

tqdm.pandas()

def findYes(row):
    row = row.iloc[0]
    quary = cl_train[(cl_train.DISPFROM <= row.I_DATE) & (cl_train.DISPEND >= row.I_DATE)]
    quary = quary[quary.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    quary['USER_ID_hash'] = row.USER_ID_hash
    return quary
    
def findNot(row):
    row = row.iloc[0]
    user_data = user_list.loc[user_list['USER_ID_hash'] == row.USER_ID_hash].iloc[0]
    quary = cl_train[(cl_train.DISPFROM <= row.I_DATE) & (cl_train.DISPEND >= row.I_DATE)]
    quary = quary[(user_data.REG_DATE <= quary.DISPEND) & (quary.DISPFROM <= user_data.WITHDRAW_DATE)]
    quary = quary[~quary.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    quary['USER_ID_hash'] = row.USER_ID_hash
    quary = quary.sample(n=9, random_state=0)
    return quary

print('Load Data')
dsdir = 'dataset/coupon-purchase-prediction'

#Dataset
cd_train = pd.read_csv(os.path.join(dsdir,'coupon_detail_train.csv'))
cl_test = pd.read_csv(os.path.join(dsdir,'coupon_list_test.csv'))
cl_train = pd.read_csv(os.path.join(dsdir,'coupon_list_train.csv'))
cv_train = pd.read_csv(os.path.join(dsdir,'coupon_visit_train.csv'))
pref_loc = pd.read_csv(os.path.join(dsdir,'prefecture_locations.csv'))
sample_sub = pd.read_csv(os.path.join(dsdir,'sample_submission.csv'))
user_list = pd.read_csv(os.path.join(dsdir,'user_list.csv'))

#Translator
pref = pd.read_csv(os.path.join(dsdir,'pref.csv'),delimiter=';',index_col='jpn')
pref_office = pd.read_csv(os.path.join(dsdir,'pref_office.csv'),delimiter=';',index_col='jpn')
small_area_name = pd.read_csv(os.path.join(dsdir,'small_area_name.csv'),delimiter=';',index_col='jpn')
big_area_name = pd.read_csv(os.path.join(dsdir,'big_area_name.csv'),delimiter=';',index_col='jpn')
capsule_text = pd.read_csv(os.path.join(dsdir,'capsule_text.csv'),delimiter=';',index_col='jpn')
genre_name = pd.read_csv(os.path.join(dsdir,'genre.csv'),delimiter=';',index_col='jpn')

print('Translate Data')
#CAPSULE TEXT
cl_test.CAPSULE_TEXT = cl_test.CAPSULE_TEXT.replace(capsule_text.to_dict()['en'])
cl_train.CAPSULE_TEXT = cl_train.CAPSULE_TEXT.replace(capsule_text.to_dict()['en'])

#GENRE NAME
cl_test.GENRE_NAME = cl_test.GENRE_NAME.replace(genre_name.to_dict()['en'])
cl_train.GENRE_NAME = cl_train.GENRE_NAME.replace(genre_name.to_dict()['en'])

#PREF NAME
cl_test.ken_name = cl_test.ken_name.replace(pref.to_dict()['en'])
cl_train.ken_name = cl_train.ken_name.replace(pref.to_dict()['en'])
pref_loc.PREF_NAME = pref_loc.PREF_NAME.replace(pref.to_dict()['en'])
user_list.PREF_NAME = user_list.PREF_NAME.replace(pref.to_dict()['en'])

#PREFECTUAL_OFFICE
pref_loc.PREFECTUAL_OFFICE = pref_loc.PREFECTUAL_OFFICE.replace(pref_office.to_dict()['en'])

#SMALL_AREA_NAME
cd_train.SMALL_AREA_NAME = cd_train.SMALL_AREA_NAME.replace(small_area_name.to_dict()['en'])
cl_test.small_area_name = cl_test.small_area_name.replace(small_area_name.to_dict()['en'])
cl_train.small_area_name = cl_train.small_area_name.replace(small_area_name.to_dict()['en'])

#large_area_name
cl_test.large_area_name = cl_test.large_area_name.replace(big_area_name.to_dict()['en'])
cl_train.large_area_name = cl_train.large_area_name.replace(big_area_name.to_dict()['en'])

print('Preprocess Data')
#cause it's annoying
cv_train.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)

cl_train.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)
cl_test.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)

cl_train['VALIDFROM'].fillna(cl_train['DISPFROM'], inplace=True)
cl_train['VALIDEND'].fillna(pd.Timestamp.max, inplace=True)

cl_test['VALIDFROM'].fillna(cl_test['DISPFROM'], inplace=True)
cl_test['VALIDEND'].fillna(pd.Timestamp.max, inplace=True)

cl_train['DISPFROM'] = pd.to_datetime(cl_train['DISPFROM'])
cl_train['DISPEND'] = pd.to_datetime(cl_train['DISPEND'])
cl_train['VALIDFROM'] = pd.to_datetime(cl_train['VALIDFROM'])
cl_train['VALIDEND'] = pd.to_datetime(cl_train['VALIDEND'])

cl_test['DISPFROM'] = pd.to_datetime(cl_test['DISPFROM'])
cl_test['DISPEND'] = pd.to_datetime(cl_test['DISPEND'])
cl_test['VALIDFROM'] = pd.to_datetime(cl_test['VALIDFROM'])
cl_test['VALIDEND'] = pd.to_datetime(cl_test['VALIDEND'])

cl_train['VALIDPERIOD'].fillna((cl_train['VALIDEND'] - cl_train['VALIDFROM'])/np.timedelta64(1,'D'), inplace=True)
cl_test['VALIDPERIOD'].fillna((cl_test['VALIDEND'] - cl_test['VALIDFROM'])/np.timedelta64(1,'D'), inplace=True)

cl_train['VALIDPERIOD'] = cl_train['VALIDPERIOD'].astype(int)
cl_test['VALIDPERIOD'] = cl_test['VALIDPERIOD'].astype(int)


cl_train.fillna(1, inplace=True)
cl_test.fillna(1, inplace=True)

user_list.WITHDRAW_DATE.fillna(pd.Timestamp.max, inplace=True)
user_list.PREF_NAME.fillna(user_list.PREF_NAME.value_counts().index[0], inplace=True)

user_list['WITHDRAW_DATE'] = pd.to_datetime(user_list['WITHDRAW_DATE'])
user_list['REG_DATE'] = pd.to_datetime(user_list['REG_DATE'])

cd_train = cd_train[['USER_ID_hash','COUPON_ID_hash','PURCHASEID_hash','I_DATE']]

print('Save CPP_REPRO_cd_train.csv')
cd_train.to_csv('CPP_REPRO_cd_train.csv',index=False)
print('Save CPP_REPRO_user_list.csv')
user_list.to_csv('CPP_REPRO_user_list.csv',index=False)

print('Create Train Data')
cvdict = cd_train[['USER_ID_hash','COUPON_ID_hash']].groupby('USER_ID_hash')['COUPON_ID_hash'].apply(list)

user_list = pd.merge(user_list,pref_loc.drop('PREFECTUAL_OFFICE', axis=1),how='left')

print('Create neg_cl_train.csv')
neg_cl_train = cd_train.groupby('PURCHASEID_hash', group_keys=False).progress_apply(findNot)
neg_cl_train = pd.merge(neg_cl_train,pref_loc.drop('PREFECTUAL_OFFICE', axis=1),how='left')
neg_cl_train['TARGET'] = 0
neg_cl_train = pd.merge(neg_cl_train, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
print('Save neg_cl_train.csv')
neg_cl_train.to_csv('neg_cl_train.csv',index=False)

print('Create pos_cl_train.csv')
pos_cl_train = cd_train.groupby('PURCHASEID_hash', group_keys=False).progress_apply(findYes)
pos_cl_train = pd.merge(pos_cl_train,pref_loc.drop('PREFECTUAL_OFFICE', axis=1),how='left')
pos_cl_train['TARGET'] = 1
pos_cl_train = pd.merge(pos_cl_train, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
print('Save pos_cl_train.csv')
pos_cl_train.to_csv('pos_cl_train.csv', index=False)

print('Combine both')
dataset = pd.concat([pos_cl_train, neg_cl_train]).reset_index(drop=True)
dataset = dataset[['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER', 'TARGET']]
dataset.sort_values('TARGET',inplace=True)
print('Drop False Negative Data')
dataset.drop_duplicates(subset=['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER'], keep='first', inplace=True)
print('Save CPP_REPRO_cl_train.csv')
dataset.to_csv('CPP_REPRO_cl_train.csv', index=False)

print('Create Test Data')
#Permutation of User-CouponTest
clist = cl_test.COUPON_ID_hash.unique().tolist()
ulist = user_list.USER_ID_hash.unique().tolist()

relations = [r for r in itertools.product(clist, ulist)]
relations = pd.DataFrame(relations,columns=['COUPON_ID_hash','USER_ID_hash'])

cl_test = pd.merge(cl_test,pref_loc.drop('PREFECTUAL_OFFICE', axis=1),how='left')
cl_test = pd.merge(relations,cl_test,how='left')
cl_test = pd.merge(cl_test, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
cl_test = cl_test[['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER']]
#cl_test = cl_test[(cl_test.DISPEND >= cl_test.REG_DATE) & (cl_test.DISPFROM <= cl_test.WITHDRAW_DATE)]
print('Save CPP_REPRO_cl_test.csv')
cl_test.to_csv('CPP_REPRO_cl_test.csv', index=False)

