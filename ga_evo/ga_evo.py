"""
# This code is ispired from the source code of:
# Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
Base64 decoder with genetic algorithm - by Nanni Bassetti - https://nannibassetti.com
"""
import random
import datetime
import sys
import time
import math
from PIL import Image
import matplotlib.pyplot as plt


geneSet = '0f'
filename="smile.raw"
with open(filename, "rb") as image2string:
    converted_string=image2string.read()
w, h = 24,24
img=Image.frombytes('RGB', (w, h), converted_string, 'raw')
imgplot = plt.imshow(img)
#plt.ion()
plt.show()
#img.show()
geneSet = '0f'
converted_string=converted_string.hex()
#print(converted_string)
#cyph = bytes(str(converted_string), 'utf-8').hex()
cyph = converted_string.encode()
i=0

def generate_parent(length):
    """
    a way to generate a random string of letters from the gene set.
    :return:
    """
    gene = []
    while len(gene) < length:
        sample_size = min(length - len(gene), len(geneSet))
        gene.extend(random.sample(geneSet, sample_size))  
    return ''.join(gene)


def get_fitness(guess):
    """
    fitness value is the total number of letters in the guess that match the letter in the same position of the message
    :param guess:
    :return: number of letters matches
    """
    # The sum() function adds the items of an iterable and returns the sum
    cguess=guess.encode()
    #print (cguess," ",cyph)
    return sum(1 for expected, actual in zip(cyph, cguess) if expected == actual)
	
def mutate(parent, minStringLen, maxStringLen):
    """
    This implementation uses an alternate replacement
    if the randomly selected newGene is the same as the one it is supposed to replace,
    which can save a significant amount of overhead.
    :param parent:
    :return:
    """
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    new_gene, alternate = random.sample(geneSet, 2)
    child_genes[index] = alternate if new_gene == child_genes[index] else new_gene

    if random.randint(0, 10) == 0:
        if len(child_genes) < maxStringLen:
            child_genes.append(random.choice(geneSet))
        elif len(child_genes) > minStringLen:
            del child_genes[random.randrange(0, len(child_genes))]

    return ''.join(child_genes)
	
def crossover(l, q):

# converting the string to list for performing the crossover
    l = list(l)
    q = list(q)
# generating the random number to perform crossover
    k = random.randint(0, len(l))
    #print("Crossover point :", k)
# interchanging the genes
    for i in range(k, min(len(l),len(q))):
        l[i], q[i] = q[i], l[i]
    l = ''.join(l)
    q = ''.join(q)
    return l

def display(guess,i):
    time_diff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    #print('{0}\t{1}\t{2}'.format(guess, fitness, time_diff))
    n = random.uniform(0.009, 0.01) # prendo una pausa del clock casuale tra 0.009 e 0.01 secondi
    time.sleep(n)
    img = Image.frombytes('RGB', (w, h), bytes.fromhex(guess), 'raw')
    imgplot = plt.imshow(img)
    #plt.ion()
    plt.show()
    sys.stdout.write('\rGeneration #'+ str(i) + '\t' + ' Fitness:' + str(fitness) + '  Time:' + str(time_diff) + " Final fitness to reach:" + str(len(cyph)))


if __name__ == '__main__':
    minStringLen = math.floor(len(cyph))
    maxStringLen = math.floor(len(cyph))
    print("len(cyph):",len(cyph),"\nmin genes:",minStringLen, "\nmax genes:",maxStringLen)
    random.seed()
    startTime = datetime.datetime.now()
    best_parent = generate_parent(minStringLen)
    best_parent2 = generate_parent(minStringLen)
    best_fitness = get_fitness(best_parent)
    #display(best_parent,i)

    while True:
        child = crossover(best_parent,best_parent2)
        child = mutate(child, minStringLen, maxStringLen)
        child_fitness = get_fitness(child)
        if best_fitness > child_fitness:
            continue
        i=i+1
        if i/10==int(i/10):
            display(child,i)
        if child_fitness == len(cyph):
            display(child,i)
            break
        best_fitness = child_fitness
        best_parent = child
        best_parent2 = mutate(child, minStringLen, maxStringLen)
        