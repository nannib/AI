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
#model = KNeighborsClassifier(n_neighbors=3)
#model = Perceptron(max_iter=60, tol=0.001, eta0=0.3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc=accuracy_score(y_test, y_pred)
valori=[74,52.524621,13.432073]
prediction = model.predict([valori])
print("Accuracy:",round(acc*100,2),"%")
print("Predicted target name: {}".format(prediction)," \n",names," \n",valori,"   ",prediction)