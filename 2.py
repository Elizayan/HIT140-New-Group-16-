import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read datasets into DataFrames
df = pd.read_csv("merged_dataset.csv")

"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""

# Create the total screentime column
total_screentime = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
df['Total_Screentime'] = df[total_screentime].sum(axis=1)

# Create the total wellbeing column
total_wellbeing = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 
                   'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df['Total_Wellbeing'] = df[total_wellbeing].sum(axis=1)

# Separate explanatory variables (x) from the response variable (y)
x = df[['Total_Screentime']].values  # Explanatory variable
y = df['Total_Wellbeing'].values  # Response variable

# Split dataset into 60% training and 40% test sets 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)  # Fixed method name
# Root Mean Square Error
rmse = math.sqrt(mse)  # Corrected here
# Normalised Root Mean Square Error
y_max = y_test.max()
y_min = y_test.min()
rmse_norm = rmse / (y_max - y_min)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)

"""
COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
VS.
A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
"""

print("\n\n##### BASELINE MODEL #####")

# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Compute standard performance metrics of the baseline model:

# Mean Absolute Error
mae_base = metrics.mean_absolute_error(y_test, y_pred_base)
# Mean Squared Error
mse_base = metrics.mean_squared_error(y_test, y_pred_base)  # Fixed method name
# Root Mean Square Error
rmse_base = math.sqrt(mse_base)  # Corrected here
# Normalised Root Mean Square Error
rmse_norm_base = rmse_base / (y_max - y_min)

print("Baseline MAE: ", mae_base)
print("Baseline MSE: ", mse_base)
print("Baseline RMSE: ", rmse_base)
print("Baseline RMSE (Normalised): ", rmse_norm_base)

