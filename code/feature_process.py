import pandas as pd

from utils.preprocessing import one_hot_encoding
from utils.load_data import load_users

import argparse

VERSION = '5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--input_data_path', default="input", type=str)
    args = parser.parse_args()
    path = args.input_data_path

    # Load raw data
    train_users, test_users = load_users(path=path)

    # Join users
    all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

    # Set ID as index
    all_users = all_users.set_index('id')
    train_users = train_users.set_index('id')
    test_users = test_users.set_index('id')

    # Drop columns
    drop_list = [
        'date_account_created',
        'date_first_active',
        'timestamp_first_active',
        'date_first_booking'
    ]

    all_users.drop(drop_list, axis=1, inplace=True)

    # IDEA: Add interaction features

    # Encode categorical features
    categorical_features = [
        'gender', 'signup_method', 'signup_flow', 'language',
        'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
        'signup_app', 'first_device_type', 'first_browser', 'most_used_device'
    ]

    all_users = one_hot_encoding(all_users, categorical_features)

    # Split into train and test users
    train_users = all_users.loc[train_users.index]
    test_users = all_users.loc[test_users.index]
    test_users.drop('country_destination', inplace=True, axis=1)

    # IDEA: Average distance to N neighbors of each class

    # Save to csv
    train_users.to_csv('preprocessed/train_users.csv')
    test_users.to_csv('preprocessed/test_users.csv')

    sessions = pd.read_csv(path + "/sessions.csv")
    categorical_features_sessions = ['action', 'action_type', 'action_detail', 'device_type']

    sessions_ohe = one_hot_encoding(sessions, categorical_features_sessions)
    sessions_ohe.to_csv('preprocessed/sessions.csv')

