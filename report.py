import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures 
import ta
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Simple EDA
# print(train_df.head())
# print(train_df.describe())
# print(test_df.head())
# print(test_df.describe())
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())

# Checking for zeroes in important columns
imp_Col = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
           'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
# print((train_df[imp_Col] == 0).sum())

# Adding lag features, moving averages, std of close price for volatility, RSI, and MACD)
train_df['lag1'] = train_df['close'].shift(1)
train_df['lag5'] = train_df['close'].shift(5)
train_df['lag10'] = train_df['close'].shift(10)

train_df['ma5'] = train_df['close'].rolling(window=5).mean()
train_df['ma10'] = train_df['close'].rolling(window=10).mean()

train_df['volatility_5'] = train_df['close'].rolling(window=5).std()
train_df['volatility_10'] = train_df['close'].rolling(window=10).std()

train_df['rsi'] = ta.momentum.RSIIndicator(train_df['close'], window=14).rsi()

macd = ta.trend.MACD(train_df['close'], window_slow=26, window_fast=12, window_sign=9)
train_df['macd'] = macd.macd()
train_df['macd_signal'] = macd.macd_signal()

# Filling missing values (if any) with 0
train_df.fillna(0, inplace=True)

# Manually adding interactions that I think are important
train_df['open_close_interaction'] = train_df['open'] * train_df['close']
train_df['volume_price_interaction'] = train_df['volume'] * train_df['close']
train_df['volatility_price_interaction'] = train_df['volatility_5'] * train_df['close']
train_df['volume_trades_interaction'] = train_df['volume'] * train_df['number_of_trades']
train_df['ma5_price_interaction'] = train_df['ma5'] * train_df['close']
train_df['ma10_price_interaction'] = train_df['ma10'] * train_df['close']
train_df['rsi_price_interaction'] = train_df['rsi'] * train_df['close']

# Setting up for Modeling
features = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
            'number_of_trades', 'lag1', 'lag5', 'lag10', 'ma5', 'ma10',
            'volatility_5', 'volatility_10', 'rsi', 'macd', 'macd_signal']

features_added = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
            'number_of_trades', 'lag1', 'lag5', 'lag10', 'ma5', 'ma10',
            'volatility_5', 'volatility_10', 'rsi', 'macd', 'macd_signal',
            'open_close_interaction', 'volume_price_interaction', 'volatility_price_interaction',
            'volume_trades_interaction', 'ma5_price_interaction', 'ma10_price_interaction',
            'rsi_price_interaction']
target = 'target'

X = train_df[features_added] # Splitting into training and validation sets
y = train_df[target]
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state = 42)

# Scaling data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)
# print(f"Training data shape: {Xtrain_scaled.shape}")
# print(f"Validation data shape: {Xval_scaled.shape}")

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
Xtrain_poly = poly.fit_transform(Xtrain_scaled)
Xval_poly = poly.transform(Xval_scaled)

np.random.seed(42)


# Model 1: Logistic Regression Model
# grid_model1 = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga'],
#     'class_weight': [None, 'balanced'],
#     'max_iter': [1000, 2000, 3000] 
# }
# grid_search = GridSearchCV(LogisticRegression(tol=1e-4, random_state=42), 
#                                               grid_model1,
#                                               scoring='f1_macro',
#                                               cv=5)
# grid_search.fit(Xtrain_poly, ytrain)
# best_model1 = grid_search.best_estimator_
# y_pred_best_model1 = best_model1.predict(Xval_poly)
# f1_best_model1 = f1_score(yval, y_pred_best_model1, average='macro')
# print(f"Best Logistic Regression F1 Score: {f1_best_model1}")
# baseline model1 = LogisticRegression(penalty='l2', 
#                             C=1.0, 
#                             solver='lbfgs',  
#                             max_iter=1000, 
#                             tol=1e-4,
#                             class_weight=None,
#                             random_state=42)
# model1.fit(Xtrain_poly, ytrain)
# y_pred_model1 = model1.predict(Xval_poly)
# f1_model1 = f1_score(yval, y_pred_model1, average='macro')
# print(f"Logistic Regression Macro F1 Score: {f1_model1}")
# F1 score around 0.4113009736103531 with all poly
# 0.40691 for l1/elasticnet and 0.401329 for manually added interactions



# Model 2: Decision Tree (dropped to improve the random forest model)
# from sklearn.tree import DecisionTreeClassifier

# model2 = DecisionTreeClassifier(random_state=42)
# model2.fit(Xtrain, ytrain)
# y_pred_model2 = model2.predict(Xval)
# f1_model2 = f1_score(yval, y_pred_model2, average='macro')
# print(f"Decision Tree Macro F1 Score: {f1_model2}")
# F1 score around 0.5051456897562353



# Model 3: Random Forest
from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
params_model3 = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt'],
}
grid_search_model3 = GridSearchCV(estimator=model3, param_grid=params_model3, cv=3, n_jobs=-1, scoring='f1', verbose=2)
grid_search_model3.fit(Xtrain_scaled, ytrain)

print(f"Best Parameters for RF: {grid_search_model3.best_params_}")

best_rf_model = grid_search_model3.best_estimator_
y_pred_rf = best_rf_model.predict(Xval_scaled)
f1_rf = f1_score(yval, y_pred_rf)
print(f"Random Forest F1 Score: {f1_rf}")
# model3.fit(Xtrain_scaled, ytrain)
# y_pred_model3 = model3.predict(Xval_scaled)
# f1_model3 = f1_score(yval, y_pred_model3, average = 'macro')
# print(f"Random Forest Macro F1 Score: {f1_model3}")
# F1 score around 0.5065548517370799

# Model 4: XGBoost
# from xgboost import XGBClassifier

# model4 = XGBClassifier(n_estimators=185, max_depth=30, random_state=42)
# model4.fit(Xtrain, ytrain)
# y_pred_model4 = model4.predict(Xval)
# f1_model4 = f1_score(yval, y_pred_model4, average='macro')
# print(f"XGBoost Macro F1 Score: {f1_model4}")
# # F1 score around 0.5045949370129568 (n=200, d=30)
# # d=50 < 40 < 29 < 33 < 35 (5044478) < 31 < 30
# # n=100 < 150 (0.504438) < 200 (0.504528348530758) < 180 

# Model 5: LightGBM
# import lightgbm as lgb

# model5 = lgb.LGBMClassifier(n_estimators=20, random_state=42)
# model5.fit(Xtrain, ytrain)
# y_pred_model5 = model5.predict(Xval)
# f1_model5 = f1_score(yval, y_pred_model5, average='macro')
# print(f"LightGBM Macro F1 Score: {f1_model5}")
# F1 score around 0.3935621955367277

# Model 6: LSTM
# steps = 10
# Xtrain_lstm = np.array([Xtrain[i:i + steps] for i in range(len(Xtrain) - steps)])
# ytrain_lstm = ytrain[steps:]
# Xval_lstm = np.array([Xval[i:i + steps] for i in range(len(Xval) - steps)])
# yval_lstm = yval[steps:]

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
# model.add(LSTM(units=50))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# # Train the model
# history = model.fit(Xtrain_lstm, ytrain_lstm, epochs=10, batch_size=64, validation_data=(Xval_lstm, yval_lstm))
# y_pred_lstm = model.predict(Xval_lstm)
# # Convert predictions to binary (0 or 1)
# y_pred_lstm_binary = np.where(y_pred_lstm > 0.5, 1, 0)
# f1_lstm = f1_score(yval_lstm, y_pred_lstm_binary, average='macro')
# print(f"LSTM Macro F1 Score: {f1_lstm}")
# F1 score around 0.3442542417131531

# Model 7: GRU
# Define the GRU model
# model = Sequential()
# model.add(GRU(units=50, return_sequences=True, input_shape=(Xtrain_lstm.shape[1], Xtrain_lstm.shape[2])))
# model.add(GRU(units=50))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# history = model.fit(Xtrain_lstm, ytrain_lstm, epochs=10, batch_size=64, validation_data=(Xval_lstm, yval_lstm))
# y_pred_GRU = model.predict(Xval_lstm)
# y_pred_GRU_bin = np.where(y_pred_GRU > 0.5, 1, 0)
# f1_GRU = f1_score(yval_lstm, y_pred_GRU_bin, average='macro')
# print(f"GRU Macro F1 Score: {f1_GRU}")
# F1 score around 0.3442587199179118

# Model 8: ANN
# Define the ANN model
# model_ann = Sequential()

# # Input layer and first hidden layer
# model_ann.add(Dense(units=64, activation='relu', input_dim=Xtrain_scaled.shape[1]))

# # Second hidden layer
# model_ann.add(Dense(units=32, activation='relu'))

# # Output layer for binary classification
# model_ann.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the ANN model
# history_ann = model_ann.fit(Xtrain_scaled, ytrain, epochs=10, batch_size=64, validation_data=(Xval_scaled, yval))

# # Predict and calculate F1 score
# y_pred_ann = model_ann.predict(Xval_scaled)
# y_pred_ann_binary = np.where(y_pred_ann > 0.5, 1, 0)
# f1_ann = f1_score(yval, y_pred_ann_binary, average='macro')
# print(f"ANN Macro F1 Score: {f1_ann}")
# F1 score around 0.3655333433043966

# Model 9: Stacked ANN
# Define the Stacked ANN model
# model_sann = Sequential()

# # Input layer and multiple hidden layers
# model_sann.add(Dense(units=128, activation='relu', input_dim=Xtrain_scaled.shape[1]))
# model_sann.add(Dense(units=64, activation='relu'))
# model_sann.add(Dense(units=32, activation='relu'))

# # Output layer for binary classification
# model_sann.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model_sann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the SANN model
# history_sann = model_sann.fit(Xtrain_scaled, ytrain, epochs=10, batch_size=64, validation_data=(Xval_scaled, yval))

# # Predict and calculate F1 score
# y_pred_sann = model_sann.predict(Xval_scaled)
# y_pred_sann_binary = np.where(y_pred_sann > 0.5, 1, 0)
# f1_sann = f1_score(yval, y_pred_sann_binary, average='macro')
# print(f"SANN Macro F1 Score: {f1_sann}")
# F1 score around 0.36324266032115426