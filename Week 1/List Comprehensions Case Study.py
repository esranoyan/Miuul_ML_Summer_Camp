# Mission 1: Capitalize and add NUM the names of the numeric variables 
# in the car_crashes data using the List Comprehension structure.

import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)

cols = ['NUM_' + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]
print(cols)

# Mission 2: Using the List Integration structure, write "FLAG" at the end of the names 
# of variables that do not have "no" in the name in the car_crashes data.

new_col = [col.upper() + "_FLAG" if 'no' not in col else col.upper() for col in df.columns]
print(new_col)

# Mission 3: Using the List Integration structure, select the names of variables that 
# DIFFER from the variable names given below and create a new dataframe.

og_list = ["abbrev", "no_previous"]

# Expected output:
"""
total  speeding  alcohol  not_distracted  ins_premium  ins_losses

0   18.8     7.332    5.640          18.048       784.55      145.08
1   18.1     7.421    4.525          16.290      1053.48      133.93
2   18.6     6.510    5.208          15.624       899.47      110.35
3   22.4     4.032    5.824          21.056       827.34      142.39
4   12.0     4.200    3.360          10.920       878.41      165.63 """

import pandas as pd
data = sns.load_dataset("car_crashes")
new_cols = [col for col in data.columns if col not in og_list]
new_data = pd.DataFrame(data = data[new_cols], columns = new_cols)
print(new_data.head())
 