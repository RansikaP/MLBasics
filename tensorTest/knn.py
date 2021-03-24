import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

labelEncoder = preprocessing.LabelEncoder()
buying = labelEncoder.fit_transform(list(data["buying"]))
maint = labelEncoder.fit_transform(list(data["maint"]))
door = labelEncoder.fit_transform(list(data["door"]))
persons = labelEncoder.fit_transform(list(data["persons"]))
lug_boot = labelEncoder.fit_transform(list(data["lug_boot"]))
safety = labelEncoder.fit_transform(list(data["safety"]))
cls = labelEncoder.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

prediction = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(prediction)):
    print("Prediction: ", names[prediction[x]], "Data: ", x_test[x], "Answer: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)