import numpy as np
import pandas as pd
import pickle
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

import argparse

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
	df_train = pd.read_csv(train_users_path)
	target = df_train['country_destination']
	df_train = df_train.drop(['country_destination'], axis=1)

	#test_users
	df_test = pd.read_csv(test_users_path)    
	id_test = df_test['id']

	#sessions
	df_sessions = pd.read_csv(sessions_path)
	df_sessions['id'] = df_sessions['user_id']
	df_sessions = df_sessions.drop(['user_id'],axis=1)