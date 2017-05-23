import pandas as pd
import numpy as np
import datetime


def load_users(path, nrows=None, na_values=np.nan):
    """Load user data."""

    train_users = path + '/train_users.csv'
    test_users = path + '/test_users.csv'

    train_users = pd.read_csv(train_users, nrows=nrows, na_values=na_values)
    test_users = pd.read_csv(test_users, nrows=nrows, na_values=na_values)

    return train_users, test_users


def load_session(path):
    sessions = pd.read_csv(path + '/sessions.csv')

    return sessions
