# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:52:54 2024

@author: nannib
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Leggi il dataset da un file CSV
df = pd.read_csv('ip2.csv', names=['date', 'time', 'ip'], header=0)
label_encoder = LabelEncoder()
# Usa LabelEncoder per convertire date e time in formato numerico
df['daten'] = label_encoder.fit_transform(df['date'])
df['timen'] = label_encoder.fit_transform(df['time']) + df['daten']

# Usa LabelEncoder per convertire l'indirizzo IP in formato numerico

df['ip_numeric'] = label_encoder.fit_transform(df['ip'])


# Seleziona le colonne rilevanti per l'Isolation Forest
X = df[['timen', 'ip_numeric']]
    
# Addestra l'Isolation Forest
feature_names = X.columns
clf = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.028),max_features=1.0)
df['is_anomaly'] = clf.fit_predict(X)



# Visualizza le righe che contengono anomalie
anomaly_rows = df[df['is_anomaly'] == -1]
print("Righe contenenti anomalie:")
print(anomaly_rows[['date', 'time', 'ip']])

# Visualizza il grafico
plt.figure(figsize=(15, 6))
plt.scatter(df['date'], df['is_anomaly'], c=df['is_anomaly'], cmap='viridis')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('date')
plt.ylabel('Anomaly Score')
plt.show()







