import pandas as pd
import numpy as np
from datetime import datetime
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures

def one_hot_encoding(data, categorical_features):
    """
    Encode a list of categorical features using a one-hot encoding.
    Param data: input Pandas Dataframe
    Param categorical_features : categorial features to encode
    Returns: a Pandas DataFrame with one-hot encoding
    """
    for feature in categorical_features:
        data_dummy = pd.get_dummies(data[feature], prefix=feature)
        data.drop([feature], axis=1, inplace=True)
        data = pd.concat((data, data_dummy), axis=1)

    return data


def process_user_actions(sessions, user):
    """Count the elapsed seconds per action.
    Parameters
    ----------
    sessions : Pandas DataFrame
        Sessions of the users.
    user : int or str
        User ID.
    Returns
    -------
    user_session_data : Series
        Returns a pandas Series with the count of each action.
    """
    # Get the user session
    user_session = sessions.loc[sessions['user_id'] == user]
    user_session_data = pd.Series()

    # Length of the session
    user_session_data['session_lenght'] = len(user_session)
    user_session_data['id'] = user

    # Take the count of each value per column
    columns = ['action', 'action_type', 'action_detail']  # device_type
    for column in columns:
        column_data = user_session[column].value_counts()
        column_data.index = column_data.index + '_count'
        user_session_data = user_session_data.append(column_data)

    # Get the most used device
    session = user_session_data.groupby(user_session_data.index).sum()

    session['most_used_device'] = user_session['device_type'].mode()
    session['most_used_device'] = np.sum(session['most_used_device'])

    if session['most_used_device'] == 0:
        session['most_used_device'] = np.nan

    # For Python 2 it's only needed to do:
    # session['most_used_device'] = user_session['device_type'].max()

    # Grouby ID and add values
    return session


def process_user_secs_elapsed(sessions, user):
    """Compute some statistical values of the elapsed seconds of a given user.
    Parameters
    ----------
    sessions : Pandas DataFrame
        Sessions of the users.
    user : int or str
        User ID.
    Returns
    -------
    user_processed_secs : Series
        Returns a pandas Series with the statistical values.
    """
    # Locate user in sessions file
    user_secs = sessions.loc[sessions['user_id'] == user, 'secs_elapsed']
    user_processed_secs = pd.Series()
    user_processed_secs['id'] = user

    user_processed_secs['secs_elapsed_sum'] = user_secs.sum()
    user_processed_secs['secs_elapsed_mean'] = user_secs.mean()
    user_processed_secs['secs_elapsed_min'] = user_secs.min()
    user_processed_secs['secs_elapsed_max'] = user_secs.max()
    user_processed_secs['secs_elapsed_quantile_1'] = user_secs.quantile(0.1)
    user_processed_secs['secs_elapsed_quantile_2'] = user_secs.quantile(0.25)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.75)
    user_processed_secs['secs_elapsed_quantile_3'] = user_secs.quantile(0.9)
    user_processed_secs['secs_elapsed_median'] = user_secs.median()
    user_processed_secs['secs_elapsed_std'] = user_secs.std()
    user_processed_secs['secs_elapsed_var'] = user_secs.var()
    user_processed_secs['secs_elapsed_skew'] = user_secs.skew()

    # Number of elapsed seconds greater than 1 day
    user_processed_secs['day_pauses'] = user_secs[user_secs > 86400].count()

    # Clicks with less than one hour of differences
    user_processed_secs['short_sessions'] = user_secs[user_secs < 3600].count()

    # Long breaks
    user_processed_secs['long_pauses'] = user_secs[user_secs > 300000].count()

    return user_processed_secs
