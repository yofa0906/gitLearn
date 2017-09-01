#python versiom
import sys
import scipy
import numpy
import matplotlib.pyplot as plt
import pandas
import time
from pandas.tools.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
start_time1 = time.time()
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(url,names=names)

elapsed_time1 = time.time() - start_time1
print(elapsed_time1)
start_time2 = time.time()
#shape
print(dataset.head(20))
elapsed_time2 = time.time() - start_time2
print(elapsed_time2)
start_time3 = time.time()
print(dataset.describe())
elapsed_time3 = time.time() - start_time3
print(elapsed_time3)

start_time4 = time.time()
print(dataset.groupby('class').size())
elapsed_time4 = time.time() - start_time4
print(elapsed_time4)

start_timew4 = time.time()
print(dataset.groupby('class').size())
elapsed_time43 = time.time() - start_timew4
print(elapsed_time43)

#boxand whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()
dataset.hist()
plt.show()

#scater plot matrix
scatter_matrix(dataset)
plt.show()
#teste