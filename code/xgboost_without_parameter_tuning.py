import argparse

from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from utils.metrics import ndcg_scorer, ndcg_score
from utils.metrics import ndcg5_score
from utils.load_data import load_users
from utils import generate_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--input_data_path', default="input", type=str)
    args = parser.parse_args()

    path = args.input_data_path
    train_users, test_users = load_users(path=path)
    train_users.fillna(-1, inplace=True)
    y_train = train_users['country_destination']
    train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
    x_train = train_users.values
    test_users_ids = test_users['id']
    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)

    xgb = XGBClassifier(
        objective="multi:softprob",
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        missing=None,
        silent=True,
        nthread=-1,
        seed=42
    )

    #kf = KFold(len(x_train), n_folds=1) # , random_state=42

    #score = cross_val_score(xgb, x_train, encoded_y_train,
    #                        cv=kf, scoring=ndcg_scorer)
    
    #print(xgb.get_params(), score.mean())
    #print (score)
    xgb.fit(x_train, encoded_y_train, eval_metric=ndcg_scorer)
    print ('@@@@@@@@@@@@@')
    predictions = xgb.predict_proba(x_train)
    print(predictions)
    print(encoded_y_train)
    score = ndcg_score(encoded_y_train, predictions)
    print (score)

    generate_submission(predictions, test_users_ids, label_encoder, name="xgboost_without_parameter_tuning")
