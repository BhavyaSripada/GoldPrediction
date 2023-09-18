
import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR 
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


st.cache_data

gold_data='gold_data.xlsx'
def load_data():
    df = pd.read_excel(gold_data,usecols='A:C',header=0, engine='openpyxl')  # Replace with your Excel file path
    return df

df = load_data()

df.columns = df.columns.str.replace(' ','')

X = df[['24karat','22karat']]  #  feature columns
y = df['24karat']  # target column


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')


X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Creating the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=42)

# Training the model on the training data with imputed features
rf_regressor.fit(X_train_imputed, y_train)



y_pred = rf_regressor.predict(X_test_imputed)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")



feature_importance = rf_regressor.feature_importances_

# Defining  feature names
feature_names = ['22 karat', '24 karat']

# Creating a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.plot(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()


# Creating line plots for '22 karat' and '24 karat'
plt.figure(figsize=(12, 6))

# 'Date' vs '22 karat' plot
plt.subplot(2, 1, 1)  # Subplot for '22 karat'
plt.plot(df.index, df['22karat'], label='22 karat', color='blue')
plt.title('Date vs 22 karat')
plt.xlabel('Date')
plt.ylabel('22 karat Price')
plt.legend()

# 'Date' vs '24 karat' plot
plt.subplot(2, 1, 2)  # Subplot for '24 karat'
plt.plot(df.index, df['24karat'], label='24 karat', color='red')
plt.title('Date vs 24 karat')
plt.xlabel('Date')
plt.ylabel('24 karat Price')
plt.legend()

plt.tight_layout()  
plt.show()


import datetime

start_date = df['Date'].max() + datetime.timedelta(days=1)
end_date = start_date + datetime.timedelta(days=540)  # 18 months = 540 days
next_18_months_dates = pd.date_range(start_date, end_date, freq='D')

# Create a DataFrame with 'Year' and 'Month' columns to use as features.
next_18_months = pd.DataFrame({
    'Year': next_18_months_dates.year,
    'Month': next_18_months_dates.month,
})

# Predict gold rates day-wise for the next 18 months using the trained model.
next_18_months['Predicted_24K_Price'] = rf_regressor.predict(next_18_months[['Year', 'Month']])

# Plot the predicted 24 karat gold rates day-wise.
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['24karat'], label='24K Gold Price (Historical)', marker='o', markersize=4)
plt.plot(next_18_months_dates, next_18_months['Predicted_24K_Price'], label='Predicted 24K Gold Price (Next 18 Months)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Predicted 24K Gold Price for the Next 18 Months (Day-wise)')
plt.grid(True)
plt.show()




from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Check stationarity using ADF test
result = adfuller(df['24karat'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


if result[1] > 0.05:
    df['Differenced_24karat'] = df['24karat'] - df['24karat'].shift(1)
    df.dropna(inplace=True)
else:
    df['Differenced_24karat'] = df['24karat']


#Plots autocorrelation and partial autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(df['Differenced_24karat'], lags=40)
plot_pacf(df['Differenced_24karat'], lags=40)
plt.show()

# Identifying the  ARIMA Order (p, d, q)
p, d, q = 1, 1, 1



model = ARIMA(df['24karat'], order=(p, d, q))
model_fit = model.fit()


# Ploting actual vs. fitted values (optional)
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['24karat'], label='Actual 24K Gold Price')
plt.plot(df['Date'], model_fit.fittedvalues, label='Fitted 24K Gold Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Fitted 24K Gold Prices')
plt.grid(True)
plt.show()



#Gold rate  Predictions for the Next 18 Months
forecast_periods = 18  

# Useing  the ARIMA model to forecast
forecast = model_fit.forecast(steps=forecast_periods)


print("Forecasted Gold Prices for the Next 18 Months:")
for i, val in enumerate(forecast):
    print(f"Month {i+1}: {val:.2f}")


last_date = df['Date'].max()
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=forecast_periods, freq='M')

# Plotting Forecasted Gold Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['24karat'], label='Historical 24K Gold Price')
plt.plot(forecast_dates, forecast, label='Forecasted 24K Gold Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Forecasted 24K Gold Prices for the Next 18 Months')
plt.grid(True)
plt.show()





