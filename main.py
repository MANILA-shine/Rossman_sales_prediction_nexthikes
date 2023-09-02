import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import os
import pickle

# Load your merged train and test data
train_data_merged = pd.read_csv("train_df.csv")
test_data_merged = pd.read_csv("test_df.csv")

train_data_merged.drop('SalesClass', axis=1, inplace=True)

train_data_merged['Date'] = pd.to_datetime(train_data_merged['Date'])
test_data_merged['Date'] = pd.to_datetime(test_data_merged['Date'])

# Define X and y
X = train_data_merged[['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                      'StoreType', 'Assortment', 'CompetitionDistance',
                      'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                      'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'weekday',
                      'is_weekend', 'Season', 'IsBeginningOfMonth',
                      'IsMidOfMonth', 'IsEndOfMonth', 'DaysToHoliday', 'DaysAfterHoliday']]
y = train_data_merged['SalesPerCustomer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Initialize MLflow
mlflow.set_experiment('Sales_Prediction')

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
with mlflow.start_run(run_name='LSTM'):
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_data=(X_test_lstm, y_test))
    y_pred = lstm_model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, y_pred)

    # Log the model to MLflow
    mlflow.keras.log_model(lstm_model, 'LSTM_model')

    # Log the parameters and metrics
    mlflow.log_params({'epochs': 10, 'batch_size': 64})
    mlflow.log_metric('mse', mse)

# Save the LSTM model using pickle
model_filename = 'lstm_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(lstm_model, f)

# Create a directory to save the models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Save the LSTM model with a timestamp
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
model_filename_with_timestamp = f'lstm_model_{timestamp}.pkl'
with open(os.path.join('saved_models', model_filename_with_timestamp), 'wb') as f:
    pickle.dump(lstm_model, f)

print('LSTM model saved successfully.')


