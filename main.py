import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from processor import data_filling, data_processing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)


def produce(m):
    x_train, y_train, x_test = data_processing()
    predictor = m.fit(x_train, y_train)
    y_predict = predictor.predict(x_test)
    np.savetxt('result.txt', y_predict, fmt='%d')


def test(m):
    x_train, y_train, x_test = data_processing()
    print(x_train.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    predictor = m.fit(x_train, y_train)
    y_predict = predictor.predict(x_test)
    rmse = mean_squared_error(y_test, y_predict) ** 0.5
    print(rmse)


# model = SVR(kernel='rbf')
# model = RandomForestRegressor(n_estimators=1000)
model = XGBRegressor(n_estimators=600, max_depth=5)

test(model)
# produce(model)
