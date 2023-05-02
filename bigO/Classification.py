import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


#classifiers = [LogisticRegression(), SVC(), DecisionTreeClassifier()]
class classifiers:
    def Evaluation(YTest, YPred, method=''):
        print('====================== ' + method + ' ======================')
        acc = accuracy_score(YTest, YPred) * 100
        print('Accuracy is %.3f%%.' % acc)
        conf = confusion_matrix(YTest, YPred)
        print(pd.DataFrame(conf, columns=['Pred-Neg', 'Pred-Pos'], index=['Actl-Neg', 'Actl-Pos']))
        precision = conf[1][1] / (conf[0][1] + conf[1][1]) if (conf[0][1] + conf[1][1]) else 0
        recall = conf[1][1] / (conf[1][0] + conf[1][1]) if (conf[1][0] + conf[1][1]) else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print('Precision is %.3f.' % precision)
        print('Recall is %.3f.' % recall)
        print('F1 score is %.3f.' % F1)
        conf = confusion_matrix(YTest, YPred)
        print('confusion_matrix :')
        print('TP = ', conf[0][0])#TP = True Positive
        print('FP = ', conf[0][1])#FP = False Positive
        print('FN = ', conf[1][0])#FN = False Negative
        print('TN = ', conf[1][1])#TN = True Negative



    @staticmethod
    def LogisticRegression(XTrain, YTrain, XTest, YTest, self=None):
        model = LogisticRegression()
        model.fit(XTrain, YTrain)
        YPred = model.predict(XTest)
        self.Evaluation(YTest, YPred, 'Logistic Regression')

    @staticmethod
    def SVM(XTrain, YTrain, XTest, YTest, self=None):
        model = SVC()
        model.fit(XTrain, YTrain)
        YPred = model.predict(XTest)
        self.Evaluation(YTest, YPred, 'SVM')

    @staticmethod
    def DecisionTree(XTrain, YTrain, XTest, YTest, self=None):
        model = DecisionTreeClassifier()
        model.fit(XTrain, YTrain)
        YPred = model.predict(XTest)
        self.Evaluation(YTest, YPred, 'Decision Tree')