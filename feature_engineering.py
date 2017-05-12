import numpy as np
import pandas as pd
import pickle
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import math

import argparse

def agebuckets(ages):
    ageless =  [i for i in range(5,101,5)] # 5, 10, 15, 20...95, 100
    buckets = ['%d-%d' %(i, i+4) for i in range(0,100,5)] # 0-4, 5-9, 10-14...90-94, 95-99
    newlist = []
    for i in range(len(ages)):
        if math.isnan(ages[i]):
            newlist.append('NA')
        elif ages[i] <ageless[0]:
            newlist.append(buckets[0])
        elif ages[i] < ageless[1]:
            newlist.append(buckets[1])
        elif ages[i] < ageless[2]:
            newlist.append(buckets[2])
        elif ages[i] < ageless[3]:
            newlist.append(buckets[3])
        elif ages[i] < ageless[4]:
            newlist.append(buckets[4])
        elif ages[i] < ageless[5]:
            newlist.append(buckets[5])
        elif ages[i] < ageless[6]:
            newlist.append(buckets[6])
        elif ages[i] < ageless[7]:
            newlist.append(buckets[7])
        elif ages[i] < ageless[8]:
            newlist.append(buckets[8])
        elif ages[i] < ageless[9]:
            newlist.append(buckets[9])
        elif ages[i] < ageless[10]:
            newlist.append(buckets[10])
        elif ages[i] < ageless[11]:
            newlist.append(buckets[11])
        elif ages[i] < ageless[12]:
            newlist.append(buckets[12]) 
        elif ages[i] < ageless[13]:
            newlist.append(buckets[13]) 
        elif ages[i] < ageless[14]:
            newlist.append(buckets[14])
        elif ages[i] < ageless[15]:
            newlist.append(buckets[15])
        elif ages[i] < ageless[16]:
            newlist.append(buckets[16])
        elif ages[i] < ageless[17]:
            newlist.append(buckets[17])
        elif ages[i] < ageless[18]:
            newlist.append(buckets[18])
        elif ages[i] < ageless[19]:
            newlist.append(buckets[19]) 
        else:
            newlist.append('100+')
    return newlist


def timedif(L1, L2):
    timediflist = []
    for i in range(len(L1)):
        try:
            if (L1[i]-L2[i]).days < 0:#datetime.timedelta(days=0):
                timediflist.append('before')
            elif (L1[i]-L2[i]).days ==0: #datetime.timedelta(days=1):
                timediflist.append('same day')
            else:
                timediflist.append('greater 1 day')
        except:
            timediflist.append('NB')
            
    return timediflist

def bookings(L1, L2, L3, L4):
    timediflist = []
    for i in range(len(L1)):
        if L1[i] == 'same day' or L2[i] == 'same day':
            timediflist.append('early')
        elif L1[i] == 'before' and L2[i] == 'before' and L3[i] == 'same day':
            timediflist.append('early')
        elif L1[i] == 'greater 1 day' and L2[i] == 'greater 1 day':
            timediflist.append('waited')
        elif L1[i] == 'greater 1 day' and L2[i] == 'before':
            timediflist.append('waited')
        elif L1[i] == 'before' and L2[i] == 'greater 1 day':
            timediflist.append('waited')
        elif L1[i] == 'before' and L2[i] == 'before' and L3[i] == 'greater 1 day':
            timediflist.append('waited')
        elif (len(L4) > 0 ) and L4[i] == 'NDF':
            timediflist.append('NB')
        else:
            timediflist.append('NA')

            
    return timediflist


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("train_user_path", help="Train data path")
	parser.add_argument("sessions_path", help="Session data path")
	parser.add_argument("test_user_path", help="Test data path")

	args = parser.parse_args()
	train_user_path = args.train_user_path
	test_user_path = args.test_user_path
	sessions_path = args.sessions_path

	print("Loading data")
	#train_users
	df_train = pd.read_csv(train_user_path)
	target = df_train['country_destination']
	#df_train = df_train.drop(['country_destination'], axis=1)

	#test_users
	df_test = pd.read_csv(test_user_path)    
	id_test = df_test['id']

	#sessions
	df_sessions = pd.read_csv(sessions_path)
	df_sessions['id'] = df_sessions['user_id']
	df_sessions = df_sessions.drop(['user_id'],axis=1)

	df_train.gender.replace("-unknown-", "NAN", inplace=True)

	df_test.gender.replace("-unknown-", "NAN", inplace=True)

	group_sessions = df_sessions.groupby("id")

	group_sessions = group_sessions.agg({'device_type':'count', 'secs_elapsed':'sum'})

	group_sessions.columns = ['sum_secs_elapsed', 'counts']

	group_sessions.reset_index(level=0, inplace=True)

	df_train.age.replace(np.nan, 0, inplace=True)
	df_train.age.astype(str, inplace=True)

	df_train.age = agebuckets(df_train.age)
	df_test.age = agebuckets(df_test.age)

	df_train.date_account_created = pd.to_datetime(df_train.date_account_created)
	df_train.timestamp_first_active = pd.to_datetime(df_train.timestamp_first_active, format = "%Y%m%d%H%M%S")
	df_train.date_first_booking = pd.to_datetime(df_train.date_first_booking)
	df_test.timestamp_first_active = pd.to_datetime(df_test.timestamp_first_active, format = "%Y%m%d%H%M%S")
	df_test.date_account_created = pd.to_datetime(df_test.date_account_created)

	#adding time lag columns
	df_train['first_book_lag'] = timedif(df_train.date_first_booking, df_train.date_account_created)
	df_train['account_active_lag'] = timedif(df_train.date_first_booking, df_train.timestamp_first_active)
	df_train['account_created_lag'] = timedif(df_train.date_account_created, df_train.timestamp_first_active)

	df_test['first_book_lag'] = "NAN"
	df_test['account_created_lag'] = timedif(df_test.date_account_created, df_test.timestamp_first_active)
	df_test['account_active_lag'] = "NAN"


	booking = bookings(df_train.first_book_lag, df_train.account_active_lag, df_train.account_created_lag, target)
	test_booking = bookings(df_test.first_book_lag, df_test.account_active_lag, df_test.account_created_lag, [])

	df_train['bookings'] = booking
	df_test["bookings"] = test_booking

	countries = pd.read_csv('input/countries.csv')
	user_demo = pd.read_csv('input/age_gender_bkts.csv')

	population_in_thous = []
	for i in range(df_train.shape[0]):
	    #print(type(df_train.loc[i, "gender"]))
	    if target[i] == 'NDF':
	        population_in_thous.append(-1) # NB = -1    
	    elif df_train.gender[i] == 'NA' or df_train.age[i] == 'NA' or df_train.loc[i, "gender"].lower() == 'nan': 
	        population_in_thous.append(-2) # NA = -2
	    elif df_train.gender[i] == 'OTHER':
	        population_in_thous.append(0)  
	    elif target[i] == 'other':
	        gendersi = user_demo.loc[user_demo.gender == df_train.loc[i, "gender"].lower(),:] 
	        ages = gendersi.loc[gendersi.age_bucket == df_train.age[i], :]
	        ages = list(map(lambda x: float(x), ages.population_in_thousands))
	        population_in_thous.append(np.mean(ages))
	    else:
	        #print(df_train.loc[i, "gender"].lower())
	        genders = user_demo.loc[user_demo.gender == df_train.loc[i, "gender"].lower(),:] 
	        dests = genders.loc[genders.country_destination == df_train.loc[i, "country_destination"], :]    
	        #print ((dests.loc[dests.age_bucket == df_train.age[i], 'population_in_thousands']))
	        population_in_thous.append(float((dests.loc[dests.age_bucket == df_train.loc[i, "age"], 'population_in_thousands'])))

	df_train['population_in_thousands'] = population_in_thous
	df_test["population_in_thousands"] = -2 # NA
	#df_train = df_train.drop(['country_destination'], axis=1)
	train_m = pd.merge(df_train, group_sessions, left_on='id', right_on ='id', how='left')
	train_m = train_m.drop('id', 1)

	#merging with grouped sessions and countries, **note most of training data is not in sessions. see below 

	test_m = pd.merge(df_test, group_sessions, left_on='id', right_on ='id', how='left')
	test_m = test_m.drop('id', 1)

	toremove = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 
              'country_destination'] 
	train_m.drop(toremove, axis=1, inplace=True)

	toremove = ['date_account_created', 'timestamp_first_active', 'date_first_booking'] 
	test_m.drop(toremove, axis=1, inplace=True)

	train_m.population_in_thousands.replace("NB", -1, inplace=True)
	train_m.population_in_thousands.replace("NA", -2, inplace=True)
	test_m.population_in_thousands.replace("NB", -1, inplace=True)
	test_m.population_in_thousands.replace("NA", -2, inplace=True)

	test_m["first_book_lag"] = "NA"
	test_m["account_active_lag"] = "NA"
	train_m.first_affiliate_tracked = train_m.first_affiliate_tracked.astype(str)

	train_m.sum_secs_elapsed.replace("nan", -2, inplace=True)
	train_m.counts.replace("nan", -2, inplace=True)
	test_m.sum_secs_elapsed.replace("nan", -2, inplace=True)
	test_m.counts.replace("nan", -2, inplace=True)

	test_m.first_affiliate_tracked = test_m.first_affiliate_tracked.astype(str)
	test_m.gender = test_m.gender.astype(str)

	from sklearn.preprocessing import OneHotEncoder
	# encode string input values as integers
	encoded_ohe_x = None
	for i in train_m:
		print ("row %s in train_m" % i)
		label_encoder = LabelEncoder()
		feature = label_encoder.fit_transform(train_m[i])
		feature = feature.reshape(train_m.shape[0], 1)
		onehot_encoder = OneHotEncoder(sparse=False)
		feature = onehot_encoder.fit_transform(feature)
		print ("one hot encoding for feature %s is:" % i)
		print (feature)
		if encoded_ohe_x is None:
			encoded_ohe_x = feature
		else:
			encoded_ohe_x = np.concatenate((encoded_ohe_x, feature), axis=1)
		print ("The one hot encoding for the features so far is:")
		print (encoded_ohe_x)
	print("X shape: : ", encoded_ohe_x.shape)

	encoded_ohe_x_test = None
	for i in test_m:
		print ("row %s in test_m" % i)
		label_encoder = LabelEncoder()
		feature = label_encoder.fit_transform(test_m[i])
		feature = feature.reshape(test_m.shape[0], 1)
		onehot_encoder = OneHotEncoder(sparse=False)
		feature = onehot_encoder.fit_transform(feature)
		if encoded_ohe_x_test is None:
			encoded_ohe_x_test = feature
		else:
			encoded_ohe_x_test = np.concatenate((encoded_ohe_x_test, feature), axis=1)
	print("X shape: : ", encoded_ohe_x_test.shape)

	le = LabelEncoder()
	le.fit(target)
	train_y = le.transform(target)

	params = {}
	params["objective"] = "multi:softmax"
	params["num_class"] = 12
	params["eta"] = 0.005
	params["min_child_weight"] = 6
	params["subsample"] = 0.7
	params["colsample_bytree"] = 0.7
	params["scale_pos_weight"] = 1
	params["silent"] = 1
	params["max_depth"] = 6
	params['eval_metric'] = "ndcg@5"
	params['nthread'] = 4

	clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
	                    objective='binary:logistic', subsample=0.5, colsample_bytree=0.5, seed=0)  

	plst = list(params.items())

	clf.fit(encoded_ohe_x, train_y)

	pred = clf.predict(encoded_ohe_x_test)

	pred = map(int,pred)
	pred = le.inverse_transform(pred)

	print set(pred)

	pred

	model.get_fscore()
