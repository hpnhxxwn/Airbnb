import sklearn.grid_search as gs
import xgboost as xgb
import numpy as np
from metrics import ndcg_scorer
from load_data import load_users
import argparse
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--input_data_path', default="input", type=str)
    args = parser.parse_args()
    path = args.input_data_path

    train_users, test_users = load_users(path=path)

    y_train = train_users['country_destination']
    train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
    train_users = train_users.fillna(-1)
    x_train = train_users.values
    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    test_users_ids = test_users['id']
    test_users.drop('id', axis=1, inplace=True)
    test_users = test_users.fillna(-1)
    x_test = test_users.values


    model = XGBClassifier()
    max_depth_values = [5, 6, 7]
    learning_rate_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    subsample_values = [0.7]
    colsample_bytree_values = [0.7]
    n_estimators = [100] #, 200
    # gamma = [0]
    params = {'max_depth' : max_depth_values, 'learning_rate': learning_rate_values, 
	          'subsample': subsample_values, 'colsample_bytree': colsample_bytree_values,
	          'n_estimators' : n_estimators 
	          #'gamma': gamma,
	          #'min_child_weight': min_child_weight
	         }
    print ("begin to do grid search to pick the best parametere")
    clf = gs.GridSearchCV(model, params, scoring=ndcg_scorer, cv=5)

    clf.fit(x_train, y_train)

    clf.grid_scores_

    clf.best_params_
