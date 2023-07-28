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

dfl = load_application_train()
print(dfl.head())

def load():
    data = pd.read_csv("titanic.csv") # a low scale dataset
    return data

dfl = load()
print(dfl.head())

##### Catching the Outlier Values ##############

# Outlier Values with Graphic #
sns.boxplot(x=dfl["Age"])
plt.show()

# How to Catch the Outlier Values #
q1 = dfl["Age"].quantile(0.25)
q3 = dfl["Age"].quantile(0.75)
print(q1,q3)

iqr = q3 - q1
print(iqr)

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr
print(up, low)

outlier_index = dfl[(dfl["Age"] < low) | (dfl["Age"] > up)].index


# Functionalization the operations above
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit, = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None): 
        return True # there is/are outlier value/s
    else:
        return False # there isn't/aren't any outlier value/s


# grab col_names #

def grab_col_names(dataframe, cat_th=10, car_th=20): # these values are optional
    """
    This function gives categoric, numeric, categoric but cardinal values's names.
    
    Parameters
    ------
    dataframe: dataframe
       Dataframe that we want to take the variable names
    cat_th: int, optional
       Class threshold value for numeric but categoric variables
    car_th: int, optional
       Class threshold value for categoric but cardinal variables
    
    Returns
    -----
    cat_cols: list
       List of categoric variables
    num_cols: list
       List of numeric variables  
    """
    
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dfl)
num_cols = [col for col in num_cols if col not in "PassengerId"]


for col in num_cols:
    print(col, check_outlier(dfl, col)) # checking the outliers for titanic dataset

cat_cols, num_cols, cat_but_car = grab_col_names(dfl)

for col in num_cols:
    print(col, check_outlier(dfl, col)) # checking the outliers for application_train dataset


# Reaching the outliers with functions #

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index     
    return outlier_index

# Removing the outliers #

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    return df_without_outliers

for col in num_cols:
    new_df = remove_outlier(dfl, col)
    
    
dfl.shape[0] - new_df[0] # shows how many outliers has been deleted

# Reassingnment with thresholds #

low, up = outlier_thresholds(dfl, "Fare")
print(dfl[(dfl["Fare"] < low) | (dfl["Fare"] > up)])

dfl.loc[(dfl["Fare"] > up), "Fare"] = up

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

dfl = load()


for col in num_cols:
    print(col, check_outlier(dfl, col))
    
for col in num_cols:
    replace_with_thresholds(dfl, col)
    
for col in num_cols:
    print(col, check_outlier(dfl, col))

# When tree methods used, we have two options:
# 1: Don't touch the outliers.
# 2: Take a small size of the outliers such as %95 to %5 or %99 to %1.
