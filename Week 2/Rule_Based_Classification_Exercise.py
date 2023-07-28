################# Output Before the Tasks #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Output After the Tasks #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


# import libraries
import pandas as pd
import seaborn as sns

# Task 1: Answer the questions below:

# Q1: Read the persona.csv file and show the general information about the dataset.
df = pd.read_csv("persona.csv")
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe().T)

# Q2: How many unique SOURCE are there? What are their frequencies?
print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())

# Q3: How many unique PRICE are there?
print(df["PRICE"].nunique())

# Q4: At what PRICE value did the sale take place?
print(df["PRICE"].value_counts())

# Q5: How many sales have been made from which COUNTRY?
print(df["COUNTRY"].value_counts())

# Q6: How much was earned in total from sales by COUNTRY?
print(df.groupby("COUNTRY").agg({"PRICE": "sum"}))

# Q7: What are the sales numbers according to the SOURCE types?
print(df["SOURCE"].value_counts())

# Q8: What are the PRICE averages by country?
print(df.groupby("COUNTRY").agg({"PRICE": "mean"}))

# Q9: What are the PRICE averages according to SOURCE's?
print(df.groupby("SOURCE").agg({"PRICE": "mean"}))

# Q10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?
print(df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"}))

# Task 2: Task 2: What is the average earnings in COUNTRY, SOURCE, SEX, AGE breakdown?
print(df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}))

#Task 3: To better see the output from the previous question, apply the sort_values method in 
# descending order according to PRICE. Save the output as agg_df.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df)

# Task 4: All variables other than PRICE included in the output of the third question are index names.
# Translate these names into variable names.
agg_df = agg_df.reset_index()
print(agg_df)

# Task 5: Convert the AGE variable to a categorical variable and add it to agg_df.
# Convert the numerical variable age to a categorical variable.
# Create December in a way that you think will be convincing.
# For Example: '0_18', '19_23', '24_30', '31_40', '41_70'

bin = [0, 18, 23, 30, 40, 66] # we split the age variable into catogories.
label = ["0_18", "19_23", "24_30", "31_40", "41_66"] # we give names to the splitted categories.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins = bin, labels = label)
print(agg_df.head())

# Task 6: Identify the new level-based customers (persona).
# Define new level-based customers (persona) and add them to the dataset as variables.
# Name of the new variable to be added as: customers_level_based
# You need to create the customers_level_based variable by combining the observations 
# in the output that you will get in the previous question.

# Attention! After creating customers_level_based values with list integration, these values must be deduplicated. 
# For example, it can be from more than one of the following expressions: USA_ANDROID_MALE_0_18. 
# It is necessary to take these into groupby and get the price averages.

agg_df.drop(["AGE", "PRICE"], axis=1).values # we dropped the columns we are not going to use.
agg_df["CUSTOMERS_LEVEL_BASED"] = ["_".join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]

agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]] # we selected the columns we are going to use.
print(agg_df.groupby("CUSTOMERS_LEVEL_BASED")["PRICE"].mean())


# Task 7: Segment new customers (personas).
# Divide new customers (for example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Add the segments to agg_df as a variable by specifying SEGMENT.
# Describe the segments (Make a group by segment and take the price mean, max, sum).
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels = ["D", "C", "B", "A"]) 
print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}))


# Task 8: Classify the newly arrived customers and estimate how much revenue they can bring.
# Which segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is she expected to earn on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user])

# Which segment does a 35-year-old French woman using IOS belong to and how much income is she expected to earn on average?
new_user_2 = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user_2])
