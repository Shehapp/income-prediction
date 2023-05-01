import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Read the CSV file
data = pd.read_csv("date set/train_data.csv")
# set all column names to lower case and delete spaces
data.columns = data.columns.str.lower().str.replace(' ', '')
data.columns = data.columns.str.lower().str.replace('-', '')

# set all column names to lower case and delete spaces
for col in data.columns:
    # if column type is object
    if data[col].dtype == 'object':
        data[col] = data[col].str.lower().str.replace(' ', '')
        data[col] = data[col].str.replace('-', '')


# fill missing values with most frequent value for loop
#set ? to nan
data = data.replace('?', np.nan)
for col in data.columns:
    data[col].fillna(data[col].mode()[0], inplace=True)



# replace values with another meaning
data=data.replace(['divorced', 'marriedafspouse',
              'marriedcivspouse', 'marriedspouseabsent',
              'nevermarried','separated','widowed'],
             ['notmarried','married','married','married',
              'notmarried','notmarried','notmarried'])



# get one hot vectors
def get_one_hot(df, column):
    #get min(10, len(df[column].unique())) most frequent values
    top_n = [x for x in df[column].value_counts().sort_values(ascending=False).head(min(10, len(df[column].unique()))).index]
    for label in top_n:
        df[column+'_'+label] = np.where(df[column]==label, 1, 0)
    df.drop(column, axis=1, inplace=True)
    return df

data = get_one_hot(data, 'workclass')
data = get_one_hot(data, 'education')
data = get_one_hot(data, 'maritalstatus')
data = get_one_hot(data, 'occupation')
data = get_one_hot(data, 'relationship')
data = get_one_hot(data, 'race')
data = get_one_hot(data, 'sex')
data = get_one_hot(data, 'nativecountry')



#Normalize numerical columns by MinMax normalization
numerical_columns = ['age', 'fnlwgt','educationnum', 'capitalgain', 'capitalloss', 'hoursperweek']
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])



# get labels
Y = data['income'].replace(['<=50k', '>50k'], [0, 1])

#get features
X = data.drop(['income'], axis=1)


print(X.columns)