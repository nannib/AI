# -*- coding: utf-8 -*-
"""

@author: Nanni Bassetti - nannibassetti.com 
"""
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier


# Load dataset
url = "cell2.csv"
names = ["power","gpsN","gpsE","cellname"]
dataset = pd.read_csv(url, names=names,header=0)
# Split-out validation dataset
dataset= pd.DataFrame(dataset)
array = dataset.values
#print (array[:,0:2])
#print(dataset.head(1000))
X = array[:,0:3]
y = array[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = SVC(gamma='auto')
#model = KNeighborsClassifier(n_neighbors=5)
#model = Perceptron(max_iter=60, tol=0.001, eta0=0.3, random_state=0)
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
acctest=accuracy_score(y_test,pred_test)
acctrain=accuracy_score(y_train,pred_train)
valori=[10, 51.41215, 12.37008]
prediction = model.predict([valori])
print("Accuracy Train:",round(acctrain*100,2),"% Accuracy Test: ",round(acctest*100,2),"%")
print("Predicted target name: {}".format(prediction)," \n",names," \n","        ",valori,"   ",prediction)
