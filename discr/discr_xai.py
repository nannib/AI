# -*- coding: utf-8 -*-
"""
This machine learning program works on a dataset forged with discrimination parameters, such the breed and the number of crimes committed in the past of the subject.
If the subject is less the 21 years old or has committed more then 2 crimes in the past and he is Martian (breed=1), the algorithm will classify him as a 'suspect'.
Otherwise, if the subject is Terrestrial (breed=1) and he has committed more then 5 crimes in the past, he will be classified as 'suspect', so Terrestrials are not influenced by the age and they need a crime edge higher then Martians, to be considered 'suspects'.
The dataset is influenced by the attributes of age and breed
@author: Nanni Bassetti - nannibassetti.com 
"""


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report
import pandas as pd
#from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import lime
import lime.lime_tabular
import numpy as np

from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

# Load dataset
url = "disct.csv"
names = ["age","breed","past_crimes","suspect"]
"""
Age
age of the subject

breed
1 - Terrestrial 
0 - Martian

past_crimes
number of crimes committed in the past
"""
df = pd.read_csv(url, names=names,header=0)
df= pd.DataFrame(df)
df.columns=['age','breed','past_crimes','suspect']



#counting the feature breed in the dataset
# and how many suspects for breed type
df1 = pd.DataFrame(df['breed'])
df2 = pd.DataFrame(df['breed'])
fig, axes = plt.subplots(nrows=1, ncols=2)
df1['breed'].value_counts().sort_values().plot(kind='bar',title='breed',rot=0,figsize=(8,9),ax=axes[0])

df2['breed'].where(df['suspect']=='yes').value_counts().sort_values().plot(kind='bar',title='suspects',rot=0,figsize=(8,9),ax=axes[1])
plt.xlabel("breed", labelpad=14)

print('Records number:',len(df),'Shape:',df.shape)
print('Suspects number:',df2['breed'].where(df['suspect']=='yes').value_counts())
print('Terrestrials:',df2['breed'].where(df['breed']=='Terrestrial').value_counts())
print('Martians:',df2['breed'].where(df['breed']=='Martian').value_counts())



#array = df.values
# values scaling
# perform a robust scaler transform of the dataset
#https://ichi.pro/it/pre-elaborazione-dei-dati-con-python-pandas-parte-3-normalizzazione-100414503679603
#df['age']=(df['age']-df['age'].mean())/df['age'].std()
#df['past_crimes']=(df['past_crimes']-df['past_crimes'].mean())/df['past_crimes'].std()
#df['breed']=(df['breed']-df['breed'].mean())/df['breed'].std()

le = preprocessing.LabelEncoder()
df['breed']=le.fit_transform(df['breed'])
#print(df['breed'])
# Split-out validation dataset
X = df[['age','breed','past_crimes']]
y = df['suspect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#model = MLPClassifier(hidden_layer_sizes=[100,50,20],verbose=2, max_iter=294, random_state=0)
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
acctest=accuracy_score(y_test,pred_test)
acctrain=accuracy_score(y_train,pred_train)


valori=[18,1,3]
prediction = model.predict([valori])

plot_confusion_matrix(model, X_train, y_train) 
plt.show() 
print(classification_report(y_train,pred_train))

#start XAI by LIME
X_featurenames = X.columns
predict_fn=lambda x:model.predict_proba(x).astype(float)
explainer=lime.lime_tabular.LimeTabularExplainer(np.array(X_train),mode='classification',feature_names= X_featurenames)
# asking for explanation for LIME model
exp = explainer.explain_instance(np.asarray(valori), predict_fn, num_features=3)
exp.as_pyplot_figure()
print(np.asarray(valori))
print("Accuracy Train:",round(acctrain*100,2),"% Accuracy Test: ",round(acctest*100,2),"%")
print (colored('CHECK A TERRESTRIAL','white','on_blue'))
print("Predicted target name: {}".format(prediction)," \n",names," \n"," ",valori,"   ",prediction)
#new_y = pd.get_dummies(df['suspect'], drop_first=True)
#print (new_y)
exp.save_to_file('lime.html')
print("Red is for Suspect NO (0), Green is for Suspect YES (1)")
#exp.show_in_notebook(show_all=False)


valori=[18,0,3]
prediction = model.predict([valori])


plot_confusion_matrix(model, X_train, y_train) 
plt.show() 
print(classification_report(y_train,pred_train))

#start XAI by LIME
X_featurenames = X.columns
predict_fn=lambda x:model.predict_proba(x).astype(float)
explainer=lime.lime_tabular.LimeTabularExplainer(np.array(X_train),mode='classification',feature_names= X_featurenames)
# asking for explanation for LIME model
exp = explainer.explain_instance(np.asarray(valori), predict_fn, num_features=3)
exp.as_pyplot_figure()

print(np.asarray(valori))
print("Accuracy Train:",round(acctrain*100,2),"% Accuracy Test: ",round(acctest*100,2),"%")
print (colored('CHECK A MARTIAN','white','on_red'))
print("Predicted target name: {}".format(prediction)," \n",names," \n"," ",valori,"   ",prediction)
#new_y = pd.get_dummies(df['suspect'], drop_first=True)
#print (new_y)
exp.save_to_file('lime.html')
print("Red is for Suspect NO (0), Green is for Suspect YES (1)")
#exp.show_in_notebook(show_all=False)
