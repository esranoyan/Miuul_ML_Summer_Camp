# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# arrange the options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# create the functions for reading the csv files
def load_application_train():
    data = pd.read_csv("application_train.csv") # a high scale dataset
    return data

df = load_application_train()


def load():
    data = pd.read_csv("titanic.csv") # a low scale dataset
    return data

df = load()

#### Standart Scaler(z) ####
# A classic standardization. x = all observations, u = mean, s = standart deviation
# z = (x - u) / s

df = load()
ss = StandardScaler()
df["Age_standart_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

##### Robust Scaler #####

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

##### Min-Max Scaler #####
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

# The scale process distribution does not distort the data it carries. 
# Only the way the data is expressed will be changed.

# Numeric to Categorical #
df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()

