import numpy as np
import pandas as pd
from bigO.preprocessing import clean_data



def getXandY(df):
    # set all column names to lower case and delete spaces from whole table symbols
    df = clean_data.clean_table_from_spaces_and_symbols(df)

    # fill missing values with most frequent value
    df = clean_data.fill_missing_values(df)

    # replace values with another meaning
    df = df.replace(['divorced', 'marriedafspouse',
                         'marriedcivspouse', 'marriedspouseabsent',
                         'nevermarried', 'separated', 'widowed'],
                        ['notmarried', 'married', 'married', 'married',
                         'notmarried', 'notmarried', 'notmarried'])

    # get one hot vectors for categorical columns
    categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex',
                           'nativecountry']
    df = clean_data.get_one_hot_all(df, categorical_columns, clean_data)

    # Normalize numerical columns by MinMax normalization
    numerical_columns = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
    df = clean_data.normalize(df, numerical_columns)

    # get labels
    Y = df['income'].replace(['<=50k', '>50k'], [0, 1])

    # get features
    X = df.drop(['income'], axis=1)
    return X, Y





# Read the CSV train file
data = pd.read_csv("date set/train_data.csv")

#get features and labels
XTrain, YTrain = getXandY(data)


# Read the CSV test file
data = pd.read_csv("date set/test_data.csv")

XTest, YTest = getXandY(data)

print("size of training data is", XTrain.shape)
print("size of testing data is", XTest.shape)