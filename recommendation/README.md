# MACHINE LEARNING - COSTRUIAMO UN CLASSIFICATORE PER LA RACCOMANDAZIONE

Tutti quelli che hanno letto il titolo e si sono fiondati a leggere l’articolo pensando di aver trovato il modo di svoltare la loro vita grazie ad un sistema automatico per esser raccomandati, rimarranno delusi.

Infatti, in questo articolo andremo a vedere come costruire, in modo semplice e didattico un piccolo programma di machine learning, che serve a prevedere e raccomandare degli articoli da acquistare, esattamente come vediamo che funzionano i grandi servizi online di musica, e-commerce, film, ecc..

Tutti sappiamo che servizi come Netflix, Spotify, Amazon, Facebook e tanti altri spesso ci suggeriscono qualcosa da vedere, ascoltare o comprare, basandosi sulle nostre scelte fatte nel passato, ma anche sulle scelte di altri utenti che hanno avuto gusti simili ai nostri.

Quindi se ad esempio ho scelto di guardare un film di fantascienza, il servizio di streaming, mi proporrà altri film di fantascienza, ma anche film di altri generi, che magari sono stati scelti da altri utenti che avevano visionato lo stesso mio film.

Questo meccanismo si chiama “filter bubbles”, ossia delle bolle di filtraggio, poiché si rimane dentro un sistema di raccomandazione filtrato attraverso delle scelte più o meno omogenee.

Ci sono tanti vantaggi in questo sistema, come la velocità, il servizio di avere qualcuno che “ti conosce” o presume di conoscerti e quindi ti serve bene, ma ci sono anche gli svantaggi dati dal “confinamento” dell’informazione, quindi si riceve sempre una stessa linea di pensiero, cosa meno grave per i film o la musica, ma potrebbe esser più grave per l’informazione politica o generale.

Fatte queste considerazioni, possiamo dare un’occhiata a come funziona un sistema di raccomandazione (recommendation), nel machine learning, che ricordiamo essere un metodo dell’Intelligenza Artificiale (IA), che utilizza modelli matematici per poter classificare o prevedere alcuni risultati, partendo da dei dati inseriti dall’uomo.

Ma andiamo subito sul pratico, quindi iniziamo parlando di un classificatore semplice ma efficace, il Nearest Neighbor, un algoritmo che classifica gli oggetti in base alla “vicinanza” ad altri oggetti, creando così la previsione per similitudine.

La vicinanza è un concetto che può essere geometrico o numerico, nel
k-Nearest Neighbors (k-NN), l’input è dato dal dataset, ossia gli oggetti inseriti sui quali il programma di machine learning dovrà addestrarsi per imparare e l’output è dato dal numero di oggetti più vicini definito dal parametro k.

Quindi, come in Figura 1, se k=3 allora il pallino verde (oggetto incognito) sarà classificato come simile ai triangoli rossi, perché i primi tre oggetti più vicini a lui sono due triangoli ed un quadrato, quindi è classificabile più come appartenente alla famiglia dei triangoli, se k=5 invece sarà classificato come più vicino ai quadrati blu, perché ci sono più quadrati, che triangoli, nelle sue vicinanze.

![k](https://github.com/nannib/AI/blob/master/recommendation/images/Immagine1.png)

Figura 1- https://commons.wikimedia.org/wiki/File:KnnClassification.svg 

La distanza che potremmo usare è quella Euclidea, ossia dati i punti [1,2,3,4] e [5,6,7,8]

La distanza è:

 ![distanza](https://github.com/nannib/AI/blob/master/recommendation/images/Cattura1.JPG)


Se applichiamo le distanze di [5,6,7,8] anche da altri punti avremo una serie di numeri e potremmo identificare quali sono quelli “meno distanti” dal nostro input per poterlo classificare.

Ma nell’esempio qui riportato useremo la distanza di Hamming, ossia

sum(abs(X1 - X2))/ len(X1)

che si calcola tenendo conto le componenti diverse di due vettori, per semplicità, se si considerano le due stringhe: ABCD e AECL, la distanza di hamming è pari a 2, cioè sono solo due le lettere diverse nelle corrispondenti posizioni.

Quindi se abbiamo [1,5,3,6] e [2,5,1,6] la distanza di hamming sarà così calcolata:

(|2-1|+|5-5|+|1-3|+|6-6|) / 4 = 0,75

Adesso abbiamo contezza del classificatore, quindi cosa ci rimane?

Dobbiamo creare il dataset iniziale, chiaramente per comodità personale, sarà tutto numerico, ricordiamoci che i computer lavorano sempre meglio con i numeri!

Dopo la creazione del dataset, scriveremo il programmino in Python, che sfrutterà l’algoritmo di k-NN e vedremo i risultati.

Il tipo d’apprendimento del machine learning utilizzato sarà “supervisionato” (Figura 2)

![apprendimento](https://github.com/nannib/AI/blob/master/recommendation/images/Immagine2.png)

Figura 2 - definizioni del tipo d'apprendimento nel machine learning
 

Iniziamo col dataset, acquisti.csv:


![dataset](https://github.com/nannib/AI/blob/master/recommendation/images/Immagine3.png)

Figura 3- dataset dei clienti C1-C21 e dei loro acquisti

Dove i codici numerici corrispondono ad i seguenti articoli:

 1   shoes

 2   gloves

 3   coffee

 4   computer

 5   book

 6   newspaper

 7   T-shirt

 8   sunglasses

I clienti sono identificati con i codici da C1 a C21, i loro acquisti passati sono nelle sei colonne successive, codificati numericamente come dalla legenda precedente, nell’ultima colonna c’è l’articolo suggerito dal sistema.

Adesso ipotizziamo un cliente C22, che acquista questi oggetti:

[6,5,0,8,1,7] ossia giornale, libro, nulla, occhiali da sole, scarpe ed una t-shirt.

Cosa consiglierà il sistema?

Ecco il risultato:

 

Accuracy Train: 78.57 % Accuracy Test:  14.29 %

Predicted target name: ['shoes'] 

 [6, 5, 0, 8, 1, 7]     ['shoes']

 

 

In questo piccolo esempio si ha un’accuratezza dell’addestramento pari al 78,57% e del testing al 14,29% quindi non si hanno fenomeni di overfitting o underfitting, ma è anche vero che stiamo parlando di un dataset veramente misero, di soli 21 elementi, non fa testo!

Se l’accuratezza fosse stata del 100% avremmo avuto un modello che si adatta male sugli input nuovi, ossia non è in grado di generalizzare bene e la forza del machine learning è saper generalizzare in modo da risolvere un problema che non si è mai palesato in precedenza, quindi trovare una soluzione nuova, se generalizza poco significa che sulle cose note va benissimo, ma di fronte a qualcosa di diverso lavorerà male.

![overunder](https://github.com/nannib/AI/blob/master/recommendation/images/Immagine4.png)

Figura 4 - tipologie di modello

  

Guardiamo il codice sorgente dello script in Python:

 

Importiamo le librerie che ci servono:

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt 

 

Carichiamo il dataset acquisti.csv

# Load dataset

url = "acquisti.csv"

dataset = pd.read_csv(url)

names = ["1 - shoes","2 - gloves","3 - coffee","4 - computer","5 - book","6 - newspaper","7 - T-shirt","8 - sunglasses"]


 

Carichiamo nella variabile X le colonne da 1 a 6 che contengono gli acquisti passati dei vari utenti e nella variabile y la settima colonna che contiene la “label”, ossia l’acquisto consigliato o l’ultimo acquisto fatto.

# calculate the Hamming distance between two vectors

def hamming_distance(a, b):

            return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)

# Split-out validation dataset

dataset = pd.read_csv(url, header=0)

array = dataset.values

print(dataset,"\n\n",names,"\n")

X = array[:,1:7]

y = array[:,7]

 

Addestriamo la macchina, considerando che il dataset è piccolo, abbiamo visto che il miglior setting per è fatto prendendo il gruppo di test pari al 30% del gruppo totale e il parametro “k” ossia n_neighbors pari a 3 punti, la distanza “hamming”.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = KNeighborsClassifier(n_neighbors = 3, metric='hamming')

model.fit(X_train, y_train)

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)

acctest=accuracy_score(y_test,pred_test)

acctrain=accuracy_score(y_train,pred_train)

 

Inseriamo nella variabile “valori” gli acquisti del nuovo cliente, non presente nel dataset

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

 

ed ecco il nostro output

Accuracy Train: 78.57 % Accuracy Test:  14.29 %

Predicted target name: ['shoes'] 

 [6, 5, 0, 8, 1, 7]     ['shoes']

 

 [6, 5, 0, 8, 1, 7]     ['shoes'] 


![plot](https://github.com/nannib/AI/blob/master/recommendation/images/Immagine5.png)

Dalla lettura del grafico, vediamo che le due distanze minori ricadono nella label “shoes”, quindi siccome due punti sono maggiori di uno e noi abbiamo impostato il k=3 ossia tre punti, l’algoritmo sceglie “shoes”.

Con questo piccolo e probabilmente bruttino programma in Python, speriamo di aver reso più chiari i meccanismi che ci sono dietro il sistema “intelligente” di raccomandazione, che ha reputato gli acquisti del nuovo cliente più vicini a tre punti del dataset ed in questi tre punti la maggioranza è formata da quelli che hanno acquistato o ricevuto come consiglio d’acquisto delle scarpe e quindi gli ha proposto la stessa cosa.

Lo stesso algoritmo e modo di ragionare sarebbe servito anche solo per classificare un input sconosciuto. That’s all folks!
