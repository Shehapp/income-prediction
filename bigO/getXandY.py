import numpy as np
import pandas as pd
from bigO.preprocessing import preprocess_data
from bigO.Feature_selection import feature_selection
from sklearn.model_selection import train_test_split

class getxandy:


    @staticmethod
    def train(df):
        # set all column names to lower case and delete spaces from whole table symbols
        df = preprocess_data.clean_table_from_spaces_and_symbols(df)

        # fill missing values with most frequent value
        df = preprocess_data.fill_missing_values(df)

        # replace values with another meaning
        df = df.replace(['divorced', 'marriedafspouse',
                         'marriedcivspouse', 'marriedspouseabsent',
                         'nevermarried', 'separated', 'widowed'],
                        ['notmarried', 'married', 'married', 'married',
                         'notmarried', 'notmarried', 'notmarried'])

        param_df = df.copy()

        # Features selection
        # get correlation between features and labels
        param_df = feature_selection.corr(param_df, feature_selection)

        drop_columns = param_df.copy()

        # drop features with correlation < 0.1
        df = df.drop(drop_columns, axis=1)

        # Normalize numerical columns by MinMax normalization
        numerical_columns = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
        # drop from numerical_columns columns with correlation < 0.1
        numerical_columns = [x for x in numerical_columns if x not in drop_columns]
        df = preprocess_data.normalize(df, numerical_columns)

        # get one hot vectors for categorical columns
        categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex',
                               'nativecountry']
        # drop from categorical_columns columns with correlation < 0.1
        categorical_columns = [x for x in categorical_columns if x not in drop_columns]
        df = preprocess_data.get_one_hot_all(df, categorical_columns, preprocess_data)

        # get labels
        Y = df['income'].replace(['<=50k', '>50k'], [0, 1])

        # get features
        X = df.drop(['income'], axis=1)
        return X, Y, drop_columns








    @staticmethod
    def test(df, drop_columns):
        # set all column names to lower case and delete spaces from whole table symbols
        df = preprocess_data.clean_table_from_spaces_and_symbols(df)

        # fill missing values with most frequent value
        df = preprocess_data.fill_missing_values(df)

        # replace values with another meaning
        df = df.replace(['divorced', 'marriedafspouse',
                         'marriedcivspouse', 'marriedspouseabsent',
                         'nevermarried', 'separated', 'widowed'],
                        ['notmarried', 'married', 'married', 'married',
                         'notmarried', 'notmarried', 'notmarried'])

        # drop features with correlation < 0.1
        df = df.drop(drop_columns, axis=1)

        # Normalize numerical columns by MinMax normalization
        numerical_columns = ['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
        # drop from numerical_columns columns with correlation < 0.1
        numerical_columns = [x for x in numerical_columns if x not in drop_columns]
        df = preprocess_data.normalize(df, numerical_columns)

        # get one hot vectors for categorical columns
        categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex',
                               'nativecountry']
        # drop from categorical_columns columns with correlation < 0.1
        categorical_columns = [x for x in categorical_columns if x not in drop_columns]
        df = preprocess_data.get_one_hot_all(df, categorical_columns, preprocess_data)

        # get labels
        Y = df['income'].replace(['<=50k', '>50k'], [0, 1])

        # get features
        X = df.drop(['income'], axis=1)
        return X, Y


