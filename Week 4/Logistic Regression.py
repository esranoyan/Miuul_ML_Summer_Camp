#### Sales Prediction with Linear Regression ####

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# read and analyze the dataset 
df = pd.read_csv("advertising.csv")
df.head()
df.shape

X = df[["TV"]]
y = df[["sales"]]

# create the model
reg_model = LinearRegression().fit(X,y)

# y_hat = b + w*x

# constant = (b - bias)
reg_model.intercept_[0] # we used the [0] notation to provide access to the value, not the array.

# w coefficient
reg_model.coef_[0][0] # we used the [0][0] notation to provide access to the value, not the array.


### Prediction ###

# Q1: If there is a tv expenditure of 150 units, how much sales would be expected ?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

## Visualize the model ##
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9},
                ci=False, color="r")
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales Numbers")
g.set_xlabel("TV Expenditure")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

## Prediction success ##

# MSE(Mean Squared Error)
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean()
y.std()

# RMSE(Root Mean Squared Error)
np.sqrt(mean_squared_error(y, y_pred))

# MAE(Mean Absolute Error)
mean_absolute_error(y, y_pred)

# R-Square
reg_model.score(X, y)


### Multiple Linear Regression ###

# read the dataset again
df = pd.read_csv("advertising.csv")

X = df.drop("sales", axis=1) # independent variable
y = df[["sales"]] # dependent variable

## Model ##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

# constant (b - bias)
reg_model.intercept_[0] # 2.9079470208164295

# Coefficients ( w - weights)
reg_model.coef_ # array([[0.0468431 , 0.17854434, 0.00258619]])


# Question : What is the expected value of the sale according to the following observation values?

# TV: 30
# radio: 10
# newspaper: 40

# Sales = b + w * x
# 2.90 + TV * 0.04 + radio * 0.17 + newspaper * 0.002
2.90 + 30 * 0.04 + 10 * 0.17 + 40 * 0.002

# how to make the same process with functionalized way
new = [[30], [10], [40]]
new = pd.DataFrame(new).T

reg_model.predict(new)


## Evaluating the Prediction Success ##

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 1.73

# Train R-Square
reg_model.score(X_train, y_train) 
# When a new variables were added, the success increased, the error decreased.

# Test RMSE 
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 1.41

# Test R-Square
reg_model.score(X_test, y_test)

# Cross Validation
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error"))) # 1.69
# This method is more reliable in this dataset.

### BONUS: Simple Linear Regression with Gradient Descent from Scratch

# Cost Function (MSE)
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    
    mse = sse / m
    return mse      

# update weights function
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
        
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w
        
# train function 
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                    cost_function(Y, initial_b, initial_w, X))) # initial error report
    
    b = initial_b
    w = initial_w
    cost_history = []
    
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        
        if i % 100 == 0:
            print("iter = {:d}   b = {:.2f}   w = {:.4f}   mse = {:.4f}".format(i, b, w, mse))
    
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters (these are optional values, you can change them according to your results)
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)