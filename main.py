import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt


df = pd.read_csv("New_York_Air_Quality.csv")
df = df.dropna(subset = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI'])

X = df[['CO','NO2','SO2','O3','PM2.5','PM10']]
Y = df['AQI']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


model = GradientBoostingRegressor(n_estimators = 100,learning_rate = 0.1,max_depth = 3,random_state=0)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

r2 = r2_score(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
rmse = sqrt(mse)

print("r2_score:",r2)
print("Mean square error:",mse)
print("Root mean square error:",rmse)
