# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np


bcancer = load_breast_cancer()
X = bcancer.data
y= bcancer.target

decisionTree= DecisionTreeClassifier(random_state=0)
randomForest = RandomForestClassifier(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 0, stratify = y)
decisionTree.fit(X_train, y_train)
randomForest.fit(X_train, y_train)

resultDecisionTree = decisionTree.predict(X_test)
resultRandomForest = randomForest.predict(X_test)

confusionDecisionTree = confusion_matrix(y_test, resultDecisionTree)
confusionRandomForest = confusion_matrix(y_test, resultRandomForest)

print('Decision Tree:')
print('Confusion matrix:')
print(confusionDecisionTree)
print('Accuracy score: '+ str(accuracy_score(resultDecisionTree,y_test)))
print('10 cross validation score: ' + str(cross_val_score(decisionTree, bcancer.data, bcancer.target, cv=10)))
print(' ')
print('Random Forest')
print('Confusion matrix:')
print(confusionRandomForest)
print('Accurary Score: '+ str(accuracy_score(resultRandomForest,y_test)))
print('10 Cross Validation Score: '+ str(cross_val_score(randomForest, bcancer.data, bcancer.target, cv=10)))


plt.matshow(confusionDecisionTree)
plt.title('Confusion matrix for Decision Tree')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(confusionRandomForest)
plt.title('Confusion matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(' ')
print('Test verisinin sınıfı:')
xtest_new = np.array([8,2,1,5,150,1,1,5,3,1,4,2,1,1,5,8,7,1,6,8,8,7,6,4,2,3,4,1,2,1])
xtest_new = xtest_new.reshape(1, -1)
test_sonuc=decisionTree.predict(xtest_new)
print(test_sonuc)
