import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")  # ignores future warnings for numpy/tensorflow
import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")  # splits the student-mat.csv attributes with ';'

data = data[["G1", "G2", "G3", "studytime", "freetime", "absences", "age"]]
# all integer attributes we want to use from student-mat.csv file

predict = "G3"  # replace G3 with any other integer value from the data list that you want this program to predict

x = np.array(data.drop([predict], 1))
# all attributes that we will use to train data, in this case; G1, G2, studytime, freetime, absences and age

y = np.array(data[predict])  # the one attribute we are trying to guess based on train data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
splits train data and test data into 4 values,
 x_train and y_train are used for making training on 90% of data that we are going to predict,
 x_test is used for predicting 10% of untrained data and y_test is the prediction x_test is trying to make
"""


""" for finding a best model(run this code at least once before commenting it out to find the highest accuracy you want)
best = 0
for _ in range(10):  # number of train models you want to use
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)  # best fit linear regression model
    acc = linear.score(x_test, y_test)  # accuracy of tests
    print(acc)

    if acc > best:
        best = acc  # saves current model if its accuracy is better than any previous train model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)  # saves to a pickle file
"""

pickle_in = open("studentmodel.pickle", "rb")  # loads the pickle file
linear = pickle.load(pickle_in)

# print("Coefficient: \n", linear.coef_)
# print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)  # predicts the x_test data based on x_train data

for x in range(len(predictions)):
    print("-"*40, "\nPrediction :", predictions[x], "\nTrain data :", x_test[x], "\nActual :", y_test[x])
"""
predictions[x] is a single prediction from the predictions list
x_test[x] are the attributes that were used to train 
y_test[x] is the actual attribute that we were trying to predict
"""

p = "G1"  # attribute which you want to compare
p2 = "G3"  # second attribute which you want to compare
style.use("ggplot")
pyplot.title("Attribute Comparison")
pyplot.scatter(data[p], data[p2])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()  # shows the plot
