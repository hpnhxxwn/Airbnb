# Airbnb new user first destination prediction
Link for the input data
```
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data
```

Make a new directory input and put all the data in this directory. Also create directories feature_engineer_processed and preprocessed.
```
mkdir input
mkdir preprocessed
mkdir feature_engineer_processed
```

## Pre-process raw data:
```
python code/feature_engineer.py  -p input
```
```
python code/feature_process.py -p input
```

### Generate model and make prediction on the test data
```
python code/xgboost_cv.py -lr 0.35 -d 5 -n 50 -ct 0.7 -sub 1 -p preprocessed
```

