# -*- coding: utf-8 -*-
"""
This machine learning program works on a dataset forged with discrimination parameters, such the breed and the number of crimes committed in the past of the subject.
If the subject is less the 21 years old or has committed more then 2 crimes in the past and he is Martian (breed=1), the algorithm will classify him as a 'suspect'.
Otherwise, if the subject is Terrestrial (breed=0) and he has committed more then 5 crimes in the past, he will be classified as 'suspect', so Terrestrials are not influenced by the age and they need a crime edge higher then Martians, to be considered 'suspects'.
The dataset is influenced by the attributes of age and breed
@author: Nanni Bassetti - nannibassetti.com 
"""


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
url = "db.csv"
names = ["age","breed","past_crimes","suspect"]
"""
Age
age of the subject

breed
0 - Terrestrial 
1 - Martian

past_crimes
number of crimes committed in the past


"""
df = pd.read_csv(url, names=names,header=0)
# Split-out validation dataset
df= pd.DataFrame(df)
array = df.values

#print (array[:,0:2])
#print(dataset.head(1000))
#sc = StandardScaler()
#sc = MinMaxScaler()
#X = sc.fit_transform(array[:,0:3])
#X = array[:,0:3]
X = df[['age','breed','past_crimes']]
#print (X)
#y = array[:,3]
y = df['suspect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#model = SVC(gamma='auto')
#model = KNeighborsClassifier(n_neighbors=3)
#model = Perceptron(max_iter=60, tol=0.001, eta0=0.3, random_state=0)
model = MLPClassifier(hidden_layer_sizes=[100,50,20],verbose=2, max_iter=294, random_state=0)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
acctest=accuracy_score(y_test,pred_test)
acctrain=accuracy_score(y_train,pred_train)
valori=[15,1,2]
prediction = model.predict([valori])

print("Accuracy Train:",round(acctrain*100,2),"% Accuracy Test: ",round(acctest*100,2),"%")
print("Predicted target name: {}".format(prediction)," \n",names," \n"," ",valori,"   ",prediction)
plot_confusion_matrix(model, X_train, y_train) 
plt.show() 
print(classification_report(y_train,pred_train))

