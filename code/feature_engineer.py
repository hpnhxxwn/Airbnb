import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import process_user_actions
from utils.preprocessing import process_user_secs_elapsed
from utils.load_data import load_users
import argparse

NROWS = None
VERSION = '5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--input_data_path', default="input", type=str)
    args = parser.parse_args()
    path = args.input_data_path

    # Load raw data
    train_users, test_users = load_users(path=path, nrows=NROWS, na_values='-unknown-')
    sessions = pd.read_csv(path + '/sessions.csv',
                           nrows=NROWS, na_values='-unknown-')

    # Join users
    all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
    all_users = all_users.set_index('id')

    # Drop date_first_booking column (empty since competition's restart)
    #all_users = all_users.drop('date_first_booking', axis=1)

    # Remove weird age values
    all_users.loc[all_users['age'] > 100, 'age'] = np.nan
    all_users.loc[all_users['age'] < 16, 'age'] = np.nan

    bins = [-1, 15, 20, 25, 30, 40, 50, 60, 75, 100]
    all_users['age_group'] = np.digitize(all_users['age'], bins, right=True)

    # Change type to date
    all_users['date_account_created'] = pd.to_datetime(
        all_users['date_account_created'], errors='ignore')
    all_users['date_first_active'] = pd.to_datetime(
        all_users['timestamp_first_active'], format='%Y%m%d%H%M%S')

    # Convert to DatetimeIndex
    date_account_created = pd.DatetimeIndex(all_users['date_account_created'])
    date_first_active = pd.DatetimeIndex(all_users['date_first_active'])
    date_first_booking = pd.DatetimeIndex(all_users['date_first_booking'])

    # Split dates into day, week, month, year
    all_users['day_account_created'] = date_account_created.day
    all_users['weekday_account_created'] = date_account_created.weekday
    all_users['week_account_created'] = date_account_created.week
    all_users['month_account_created'] = date_account_created.month
    all_users['year_account_created'] = date_account_created.year
    all_users['day_first_active'] = date_first_active.day
    all_users['weekday_first_active'] = date_first_active.weekday
    all_users['week_first_active'] = date_first_active.week
    all_users['month_first_active'] = date_first_active.month
    all_users['year_first_active'] = date_first_active.year

    all_users['day_first_book'] = date_first_booking.day
    all_users['weekday_first_book'] = date_first_booking.weekday
    all_users['week_first_book'] = date_first_booking.week
    all_users['month_first_book'] = date_first_booking.month
    all_users['year_first_book'] = date_first_booking.year

    # IDEA: Classify and group by dispositive

    # Get the count of general session information
    result = sessions.groupby('user_id').count()
    result.rename(columns=lambda x: x + '_count', inplace=True)
    all_users = pd.concat([all_users, result], axis=1)

    # Add number of NaNs per row
    all_users['nan_sum'] = all_users.isnull().sum(axis=1)

    # To improve performance we translate each different user_id string into a
    # integer. This yields almost 50% of performance gain when multiprocessing
    # because Python pickles strings and integers differently
    le = LabelEncoder()
    sessions['user_id'] = le.fit_transform(sessions['user_id'].astype(str))
    sessions_ids = sessions['user_id'].unique()

    # Make pool to process in parallel
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    # Count of each user action in sessions
    result = p.map(partial(process_user_actions, sessions), sessions_ids)
    result = pd.DataFrame(result)
    result['id'] = le.inverse_transform(result['id'])
    all_users = pd.concat([all_users, result.set_index('id')], axis=1)

    print(all_users.most_used_device.value_counts())
    # Elapsed seconds statistics
    result = p.map(partial(process_user_secs_elapsed, sessions), sessions_ids)
    result = pd.DataFrame(result).set_index('id')
    result.index = le.inverse_transform(result.index.values.astype(int))
    all_users = pd.concat([all_users, result], axis=1)

    # Set ID as index
    train_users = train_users.set_index('id')
    test_users = test_users.set_index('id')

    # Split into train and test users
    all_users.index.name = 'id'
    processed_train_users = all_users.loc[train_users.index]
    processed_test_users = all_users.loc[test_users.index]
    processed_test_users.drop(['country_destination'], inplace=True, axis=1)

    # Save to csv
    processed_train_users.to_csv('feature_engineer_processed/train_users.csv')
    processed_test_users.to_csv('feature_engineer_processed/test_users.csv')
