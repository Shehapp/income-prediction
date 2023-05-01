import numpy as np
from sklearn.preprocessing import MinMaxScaler



class clean_data:

     # set all column names to lower case and delete spaces from whole table symbols
     @staticmethod
     def clean_table_from_spaces_and_symbols(df):
         # clean column names
         df.columns = df.columns.str.lower().str.replace(' ', '')
         df.columns = df.columns.str.lower().str.replace('-', '')
         # clean values
         for col in df.columns:
             # if column type is object
             if df[col].dtype == 'object':
                 df[col] = df[col].str.lower().str.replace(' ', '')
                 df[col] = df[col].str.replace('-', '')
         return df



     # fill missing values with most frequent value
     @staticmethod
     def fill_missing_values(df):
         df = df.replace('?', np.nan)
         for col in df.columns:
             df[col].fillna(df[col].mode()[0], inplace=True)
         return df



     # get one hot vectors
     @staticmethod
     def get_one_hot(df, column):
         # get min(10, len(df[column].unique())) most frequent values
         top_n = [x for x in
                  df[column].value_counts().sort_values(ascending=False).head(min(10, len(df[column].unique()))).index]
         for label in top_n:
             df[column + '_' + label] = np.where(df[column] == label, 1, 0)
         df.drop(column, axis=1, inplace=True)
         return df



     # get one hot vectors for categorical columns
     @staticmethod
     def get_one_hot_all(df, categorical_columns, self=None):
         for column in categorical_columns:
             df = self.get_one_hot(df, column)
         return df



     # Normalize numerical columns by MinMax normalization
     @staticmethod
     def normalize(df, numerical_columns):
         scaler = MinMaxScaler()
         df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
         return df