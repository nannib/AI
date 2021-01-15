from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Load dataset
url = "acquisti.csv"
# 1	shoes
# 2	gloves
# 3	coffee
# 4	computer
# 5	book
# 6	newspaper
# 7	T-shirt
# 8	sunglasses


# calculate the Hamming distance between two vectors
def hamming_distance(a, b):
	return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)


names = ["1 - shoes","2 - gloves","3 - coffee","4 - computer","5 - book","6 - newspaper","7 - T-shirt","8 - sunglasses"]
dataset = pd.read_csv(url, header=0)
# Split-out validation dataset
dataset= pd.DataFrame(dataset)
array = dataset.values
print(dataset,"\n\n",names,"\n")
X = array[:,1:7]
y = array[:,7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = KNeighborsClassifier(n_neighbors = 3, metric='hamming')
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
acctest=accuracy_score(y_test,pred_test)
acctrain=accuracy_score(y_train,pred_train)
valori=[6,5,0,8,1,7] 
i=0 
ds = [] 
for row in X:
    i=i+1
    distance = hamming_distance(valori, row)
    print(row," ",round(distance,2)," ",y[i-1])
    ds.append(distance)
#print(ds)
plt.plot(ds, y,'ro')
plt.show()
prediction = model.predict([valori])
print("Accuracy Train:",round(acctrain*100,2),"% Accuracy Test: ",round(acctest*100,2),"%")
print("Predicted target name: {}".format(prediction)," \n\n",valori,"   ",prediction)

