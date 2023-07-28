# import libraries
import pandas as pd
import seaborn as sns

# Mission 1: Describe the Titanic dataset from the Seaborn library.
df = sns.load_dataset("titanic")
# print(df.head())

# Mission 2: Find the number of the male and female passengers in the Titanic dataset.
male_number = df["sex"].value_counts()["male"]
female_number = df["sex"].value_counts()["female"]
print("Male number: {} \nFemale number: {}".format(male_number, female_number))

# Mission 3: Find the number of unique values for each column.
print(df.nunique())

# Mission 4: Find the number of unique values of the pclass variable.
print(df["pclass"].nunique())

# Mission 5: Find the number of unique values of the pclass and parch variables.
print(df[["pclass", "parch"]].nunique())

# Mission 6: Check the type of the embarked variable. Change the type to category and check again
print(df["embarked"].dtype) # output: object
df["embarked"] = df["embarked"].astype("category") 
print(df["embarked"].dtype) #output: category

# Mission 7: Show all the information of those with embarked value is C.
print(df[df["embarked"] == "C"].head())

# Mission 8: Show all the information of those who do not have an embarked value is S.
print(df[df["embarked"] != "S"].head())

# Mission 9: Show all the information of passengers who is female and under the age of 30.
print(df[(df["sex"] == "female") & (df["age"] < 30)].head())

# Mission 10: Please show the information of passengers whose Fare's older than 500 or older than 70 years old.
print(df[(df["fare"] > 500) | (df["age"] > 70)].head())

# Mission 11: Find the sum of the null values in each variable.
print(df.isnull().sum())

# Mission 12: Extract the who variable from the dataframe.
print(df.drop("who", axis=1, inplace=True))

# Mission 13: Fill in the empty values in the deck variable with the most repeating value (mode) of the deck variable.
print(df["deck"].fillna(df["deck"].mode()[0], inplace=True)) 
# the output gives a serie(mode can be more than one variable), so mode()[0] is for selecting the first index as the mode.

# Mission 14: Fill in the empty values in the age variable with the median of the age variable.
print(df["age"].fillna(df["age"].median(), inplace=True))

# Mission 15: Find the sum, count, mean values in the breakdown of the pclass and sex variables of the survived variable.
print(df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]}))

# Mission 16: Write a function that will give 1 to those under the age of 30 and will give 0 to those above the age of 30 or equal to 30. 
# Use the function you created and create a variable named age_flag in the Titanic dataset. (use the apply and lambda structures)  
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)


# Mission 17: Describe the Tips dataset from the Seaborn library.
df2 = sns.load_dataset("tips")
print(df2.head())

# Mission 18: Find the sum, min, max and average of the total bill values according to the categories (Dinner, Lunch) of the time variable.
print(df2.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]}))

# Mission 19: Find the sum, min, max and average of the total_bill values according to the days and time.
print(df2.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]}))

# Mission 20: Find the sum, min, max and average of the total_bill and tip values for lunch time and female customers according to the day.
data = df2[(df2['time'] == 'Lunch') & (df2['sex'] == 'Female')]
print(data.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                               "tip": ["sum", "min", "max", "mean"]}))

# Mission 21: What is the average of orders with size less than 3 and total_bill greater than 10? (use loc)
print(df2.loc[(df2["size"] < 3) & (df2["total_bill"] > 10), "total_bill"].mean())
 # I got an error says : Select only valid columns or specify the value of numeric_only to silence this warning. That's why I added "total_bill" at the end.
 
 # Mission 22: Create a new variable named total_bill_tip_sum and give the total amount of bills and tips paid by each customer.
df2["total_bill_tip_sum"] = df2["tip"] + df2["total_bill"]
print(df2.head())

# Mission 23: Sort from large to small according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe.
new_df2 = df2.sort_values("total_bill_tip_sum", ascending = False).head(30)
print(new_df2)

