import pandas as pd
from sklearn.model_selection import train_test_split
from bigO.getXandY import getxandy
from bigO.Classification import classifiers


#read data
train_data = pd.read_csv("date set/train_data.csv")
test_data = pd.read_csv("date set/test_data.csv")



#test model by test_data
'''
XTrain, YTrain, drop_columns  = getxandy.train(train_data)
XTest, YTest = getxandy.test(test_data, drop_columns)
'''

#test model by train_data
# separate training data and validation data.
X, Y, drop_columns = getxandy.train(train_data)
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)


print("size of training data is", XTrain.shape)
print("size of testing data is", XTest.shape)

print(XTrain.columns)




# Evaluate the model.
classifiers.DecisionTree(XTrain, YTrain, XTest, YTest, classifiers)


