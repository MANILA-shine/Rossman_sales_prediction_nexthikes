import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

train_data_merged=pd.read_csv("C:\Users\manil\ROSSMAN_SALES_PREDICTION_BY_MANILA\train_df.csv")
test_data_merged=pd.read_csv("C:\Users\manil\ROSSMAN_SALES_PREDICTION_BY_MANILA\test_df.csv")

train_data_merged.info()

test_data_merged['Date'] = pd.to_datetime(test_data_merged['Date'])

train_data_merged['Date'] = pd.to_datetime(train_data_merged['Date'])

train_data_merged.drop('SalesClass', axis=1, inplace=True)

train_data_merged.info()

test_data_merged.info()



import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import pickle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define transformers for numeric features
numeric_features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define models
models = [
    ('LR', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('SVM', SVR(kernel='linear')),
    ('KNN', KNeighborsRegressor(n_neighbors=5)),
    ('Neural Network', MLPRegressor(max_iter=1000, random_state=42)),
    ('LSTM', None)  # To be added later
]

# Define the LSTM model
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
# Initialize MLflow
mlflow.set_experiment('Sales_Prediction')

 # Create pipelines and calculate MSE
results = []
for name, model in models:
    with mlflow.start_run(run_name=name):
        if name == 'LSTM':
            lstm_model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
            lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
        else:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results.append((name, mse))

        # Log the model to MLflow
        mlflow.sklearn.log_model(pipeline, name)

        # Log the parameters and metrics
        mlflow.log_params(pipeline.named_steps['model'].get_params())
        mlflow.log_metric('mse', mse)

# Create a DataFrame to compare the results
results_df = pd.DataFrame(results, columns=['Model', 'MSE'])

   # Save models using pickle
if name == 'LSTM':
    with open(os.path.join('saved_models', f'{name}_model.pkl'), 'wb') as f:
        pickle.dump(lstm_model, f)
else:
    filename = os.path.join('saved_models', f'{name}_model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    mlflow.sklearn.log_model(model, f'{name}_model')
    print(f'Saved {name} model to {filename}')


# Save results DataFrame to CSV
results_df.to_csv('results.csv', index=False)
print('Models and results saved successfully.')

# Create a directory to save the models
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Serialize and save models with timestamps
for name, model in models:
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')
    model_filename = f'{name}_{timestamp}.pkl'

    if name == 'LSTM':
        lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)

        with open(os.path.join('saved_models', model_filename), 'wb') as f:
            pickle.dump(lstm_model, f)
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)

        with open(os.path.join('saved_models', model_filename), 'wb') as f:
            pickle.dump(pipeline, f)

    print(f'Saved {name} model with timestamp {timestamp}')

print('Models saved successfully.')

