# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:42:49 2021

@author: nanni bassetti - nannibassetti.com
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


df = pd.read_csv('ip.csv', names=['date', 'time', 'ip'], header=0)

F = df[['time','ip']]

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.05),max_features=1.0)
model.fit(F)
df['scores']=model.decision_function(F)
df['anomaly']=model.fit_predict(F)

anomaly=df.loc[df['anomaly']==-1]
normal=df.loc[df['anomaly'] == 1]

anomaly_index=list(anomaly.index)
outliers_counter = len(df[df['anomaly'] < 0])

print(anomaly)
print ('Anomalies number: ',outliers_counter)
print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))

p1 = plt.scatter(normal, normal, c="green", s=50, edgecolor="black")
p2 = plt.scatter(anomaly, anomaly, c="red", s=50, edgecolor="black")
plt.xlim((-0.3, 0.3))
plt.ylim((-0.3, 0.3))
plt.legend(
    [p1, p2],
    ["normal", "anomalous"],
    loc="lower right",
)

plt.show()