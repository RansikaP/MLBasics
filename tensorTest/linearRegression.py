import pickle

import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

with open("studentmodel.pickle", "rb") as f:
    linear = pickle.load(f)

print("Co: ", linear.coef_)
print("Intercept: \n", linear.intercept_ )

predictions = linear.predict(x_test)

for i in range (len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
