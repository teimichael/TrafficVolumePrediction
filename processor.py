import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def data_filling():
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    train_data = fill_holiday(train_data)
    test_data = fill_holiday(test_data)

    train_data.to_csv('./processed/train_data_filled.csv', index=False)
    test_data.to_csv('./processed/test_data_filled.csv', index=False)


def fill_holiday(data):
    for i in range(len(data)):
        if data.loc[i, 'holiday'] != 'None':
            holiday_list = [data.loc[i, 'holiday']] * len(
                data.loc[data['timestamp'].str.contains(data.loc[i, 'timestamp'].split(' ')[0])])
            data.loc[data['timestamp'].str.contains(
                data.loc[i, 'timestamp'].split(' ')[0]), 'holiday'] = holiday_list
    return data


def data_processing():
    train_data = pd.read_csv("processed/train_data_filled.csv")
    test_data = pd.read_csv("processed/test_data_filled.csv")

    x_train = train_data.drop('traffic_volume', axis=1)
    y_train = train_data['traffic_volume']
    x_test = test_data
    all_data = pd.concat([x_train, x_test])

    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
    all_data['weekday'] = all_data['timestamp'].dt.weekday
    all_data['month'] = all_data['timestamp'].dt.month
    # all_data['day_of_month'] = all_data['timestamp'].dt.day
    all_data['hour'] = all_data['timestamp'].dt.hour
    all_data = all_data.drop(['timestamp'], axis=1)

    holiday_dummy = pd.get_dummies(all_data['holiday'])
    weather_dummy = pd.get_dummies(all_data['weather'])
    weather_detail_dummy = pd.get_dummies(all_data['weather_detail'])

    all_data.drop('holiday', axis=1, inplace=True)
    all_data.drop('weather', axis=1, inplace=True)
    all_data.drop('weather_detail', axis=1, inplace=True)

    scaler = StandardScaler()
    scaler.fit(all_data)
    all_data = scaler.transform(all_data)

    all_data = np.hstack((all_data, holiday_dummy, weather_dummy, weather_detail_dummy))

    x_train = all_data[:len(x_train)]
    x_test = all_data[len(x_train):]
    return x_train, y_train, x_test
