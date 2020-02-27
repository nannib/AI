# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:16:42 2020

@author: nannib
"""

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url = "nb.csv"
names = ["hour","email"]
dataset = pd.read_csv(url, names=names,header=0)
# Split-out validation dataset
dataset= pd.DataFrame(dataset)
array = dataset.values
#print (array[:,0:2])
#print(dataset.head(1000))
X = array[:,:-1]
y = array[:, 1]
#print (X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Fitting Simple Linear Regression to the Training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict([[100]])
print (y_pred)
# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('E-mail echanged (Training set)')
viz_train.xlabel('Working hours')
viz_train.ylabel('Number of e-mails')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('E-mail echanged (Training set)')
viz_train.xlabel('Working hours')
viz_train.ylabel('Number of e-mails')
viz_test.show()