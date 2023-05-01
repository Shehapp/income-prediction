import numpy as np
import pandas as pd
from bigO.preprocessing import clean_data


# Read the CSV file
data = pd.read_csv("date set/train_data.csv")


# set all column names to lower case and delete spaces from whole table symbols
data = clean_data.clean_table_from_spaces_and_symbols(data)


# fill missing values with most frequent value
data = clean_data.fill_missing_values(data)



# replace values with another meaning
data=data.replace(['divorced', 'marriedafspouse',
              'marriedcivspouse', 'marriedspouseabsent',
              'nevermarried','separated','widowed'],
             ['notmarried','married','married','married',
              'notmarried','notmarried','notmarried'])



# get one hot vectors for categorical columns
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex','nativecountry']
data = clean_data.get_one_hot_all(data, categorical_columns, clean_data)



#Normalize numerical columns by MinMax normalization
numerical_columns = ['age', 'fnlwgt','educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
data = clean_data.normalize(data, numerical_columns)


# get labels
Y = data['income'].replace(['<=50k', '>50k'], [0, 1])

#get features
X = data.drop(['income'], axis=1)

print(X.head())