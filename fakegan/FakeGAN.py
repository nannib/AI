# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:31:16 2023
FakeGan for educational purpose.
@author: Nanni Bassetti - nannibassetti.com
"""
import random
import math
from sys import exit

random_str=""
distance=-1

# Funzione per generare stringhe casuali di 6 lettere
def generator(random_str,distance):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if distance==-1:
        lv=''.join(random.choice(alphabet) for i in range(6))
# la prima stringa casuale generata è modificata sostituendo dei caratteri presi dall'alfabeto, così la stringa si evolve per raggiungere la distanza che il discriminatore accetterà.
    if distance > 0:
        index = random.randint(0, 5)
        lv = random_str[:index] + random.choice(alphabet.replace(random_str[index], '')) + random_str[index+1:]
    return lv

# Funzione per discriminare le stringhe
def discriminator(random_str, file_name):
    with open(file_name, "r") as f:
        ab_strings = f.read().splitlines()
        
    for ab_str in ab_strings:
        # Calcola la distanza euclidea tra la stringa casuale e quella nel file
        distance=math.sqrt(sum([(ord(ab_str[i])-ord(random_str[i]))**2 for i in range(6)]))
        # Se la distanza è compresa tra 0 e 2 esclusi, stampa la stringa casuale e esce dalla funzione
        if distance > 0 and distance < 2:
            print("stringa generata: ",random_str," distance:",distance,"stringa reale:",ab_str)
            exit()
    return distance

# Esempio di utilizzo delle funzioni

for i in range(50000): # il numero 50000 è arbitrario
        random_str = generator(random_str,distance)
        distance=discriminator(random_str, "ab.txt")
    
