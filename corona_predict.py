# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:16:42 2020

@author: Nanni Bassetti - nannibassetti.com 
"""

# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url = "corona.csv"
names = ["city","gpsN","gpsE","infected"]
dataset = pd.read_csv(url, names=names,header=0)
# Split-out validation dataset
dataset= pd.DataFrame(dataset)
array = dataset.values
#print (array[:,1:3])
print(dataset.head(1000))
X = array[:,1:3]
y = array[:,3]
#print (X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Fitting Simple Linear Regression to the Training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)
inputgpsN=45.28081605
inputgpsE=9.27355964
# Predicting the Test set results
y_pred = regressor.predict([[inputgpsN,inputgpsE]])
print ("\n Prediction of number of infected: ",y_pred, " at Gps coords:", inputgpsN," ",inputgpsE )

