import pandas as pd

import sys
import os

if '__file__' in globals():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    parent_dir = os.getcwd()
sys.path.insert(0, parent_dir)

from lib.OneRule import OneRuleClassifier
from lib.NaiveBayes import NaiveBayesClassifier
from lib.DecisionTree import DecisionTreeClassifier
from lib.KNN import KNNClassifier

file_path = os.getcwd() + '\\Classification\\res\\PlayTest\\train.csv'
df = pd.read_csv(file_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

classifiers = [(OneRuleClassifier(), '1R-my'),
        (NaiveBayesClassifier(), 'NB-my'),
        (DecisionTreeClassifier(), 'DT-my'),
        (KNNClassifier(k=3, weights=False), 'KNN-my')]

for (clf, name) in classifiers:
    clf.fit(X, y)

file_path = os.getcwd() + '\\Classification\\res\\PlayTest\\test.csv'
df = pd.read_csv(file_path)
X_test = df.iloc[:, :-1]

for (clf, name) in classifiers:
    res = clf.predict(X_test)
    print("Model " + name + " predicted " + str(res[0]))