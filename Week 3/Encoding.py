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


############ Label Encoding & Binary Encoding #####################
# The purpose of label encoding is to adapt the data to the standards that algorithms expect from us in machine learning.
dff = load()
dff.head()
dff["Sex"].head()

le = LabelEncoder()
le.fit_transform(dff["Sex"])[0:5]

le.inverse_transform([0,1]) # this shows which value describes which value, you can use it when you forget the values.
# female = 0 , male = 1

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

dff = load()

binary_cols = [col for col in dff.columns if dff[col].dtype not in ["int64", "float64"]
               and dff[col].nunique() == 2] # if we have a high scale dataset this is the way to choose categorical variables.

for col in binary_cols:
    label_encoder(dff, col)
dff.head()

# do the same things in the high scale dataset
df = load_application_train()

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)


#### One-Hot Encoding ######

dff = load()
dff.head()
dff["Embarked"].value_counts()

pd.get_dummies(dff, columns=["Embarked"]).head()

pd.get_dummies(dff,columns=["Embarked"],drop_first=True).head()

pd.get_dummies(dff,columns=["Embarked"],dummy_na=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = load()

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

one_hot_encoder(dff, ohe_cols).head()

######### Rare Encoding ############
 
# First Step: Analyzing the minority-multiplicity state of categorical variables
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
        
for col in cat_cols:
    cat_summary(df,col)
    
# Second Step: Analyzing the relationship between the categories in the minority and the dependent variable
df["NAME_EDUCATION_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyzer(df, "TARGET", cat_cols)

# Third Step: Creating the rare encoder function

def rare_encoder(dataframe, rare_perc):
    temp_df: dataframe.copy()
    
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
        
    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyzer(new_df, "TARGET", cat_cols)
df["OCCUPATION_TYPE"].value_counts()