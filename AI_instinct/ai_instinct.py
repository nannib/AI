# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:39:30 2024

@author: nannib
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import json

# Definisci la funzione di fitness basata su distanza + coordinazione
def swimming_efficiency(leg_movements):
    """
    Il fitness è determinato da quanta distanza in direzione x
    il quadrupede può percorrere in base ai movimenti delle sue gambe
    e quanto sono coordinati i movimenti.
    """
    num_legs = 4
    body_center = np.array([0, 0])  # Centro del corpo del quadrupede all'origine
    leg_positions = np.zeros((num_legs, 2))  # Memorizza le posizioni finali (x, y) di ogni gamba

    # Punti di partenza fissi per ogni gamba (punti di attacco sul corpo)
    start_positions = [
        np.array([-1, 1]),  # Gamba anteriore sinistra
        np.array([1, 1]),   # Gamba anteriore destra
        np.array([-1, -1]), # Gamba posteriore sinistra
        np.array([1, -1])   # Gamba posteriore destra
    ]

    distances = np.zeros(num_legs)  # Distanza che ogni gamba si sposta in direzione x
    total_distance = 0
    coordination_score = 0

    # Calcola le posizioni finali delle gambe in base all'ampiezza (lunghezza) e all'angolo
    for i in range(num_legs):
        length, angle = leg_movements[i]
        leg_positions[i] = start_positions[i] + np.array([length * np.cos(angle), length * np.sin(angle)])
        
        # Calcola la distanza percorsa da ogni gamba in direzione x
        distances[i] = leg_positions[i][0] - start_positions[i][0]

    # Calcola la coordinazione (ricompensa i movimenti alternati: simmetria sinistra-destra)
    coordination_score = np.abs(distances[0] - distances[1]) + np.abs(distances[2] - distances[3])

    # Distanza totale percorsa in avanti
    total_distance = np.sum(distances)

    # Combina distanza e coordinazione in un punteggio di fitness
    fitness = total_distance - 0.5 * coordination_score  # Penalizziamo la scarsa coordinazione
    return fitness

# Genera una popolazione iniziale di movimenti delle gambe casuali
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = []
        for _ in range(4):  # Quattro gambe
            length = random.uniform(0.5, 2.0)  # Lunghezza tra 0.5 e 2.0
            angle = random.uniform(0, 2 * np.pi)  # Angolo tra 0 e 2*pi
            individual.append([length, angle])
        population.append(individual)
    return population

# Funzione di selezione (roulette wheel)
def select_parents(population, fitnesses):
    # Assicurati che tutti i valori di fitness siano non negativi aggiungendo una costante se necessario
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        fitnesses = [f - min_fitness for f in fitnesses]  # Sposta tutti i fitness per renderli non negativi
    
    total_fitness = sum(fitnesses)

    if total_fitness == 0:
        # Se tutti i fitness sono zero, seleziona i genitori casualmente
        selection_probs = [1 / len(fitnesses)] * len(fitnesses)
    else:
        # Normalizza i valori di fitness per creare le probabilità di selezione
        selection_probs = [f / total_fitness for f in fitnesses]

    # Scegli due genitori in base alle probabilità di selezione
    parent_indices = np.random.choice(len(population), size=2, p=selection_probs)
    parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
    
    return parent1, parent2

# Crossover (crossover blend tra due genitori)
def crossover(parent1, parent2):
    child = []
    alpha = random.uniform(0, 1)
    for leg in range(4):
        leg_movement = [
            alpha * parent1[leg][0] + (1 - alpha) * parent2[leg][0],  # Lunghezza
            alpha * parent1[leg][1] + (1 - alpha) * parent2[leg][1]   # Angolo
        ]
        child.append(leg_movement)
    return child

# Mutazione (perturba casualmente il movimento di una gamba)
def mutate(individual, mutation_rate=0.1):
    for leg in range(4):
        if random.random() < mutation_rate:
            individual[leg][0] += random.uniform(-0.1, 0.1)  # Mutazione sulla lunghezza
            individual[leg][1] += random.uniform(-0.1, 0.1)  # Mutazione sull'angolo
    return individual

# Salva la migliore soluzione in un file
def save_solution(solution, filename="best_swimming_solution.json"):
    with open(filename, "w") as f:
        json.dump(solution, f)

# Traccia i movimenti delle gambe come segmenti
def plot_leg_movements(leg_movements, generation):
    # Punti di partenza fissi per ogni gamba
    start_positions = [
        np.array([-1, 1]),  # Gamba anteriore sinistra
        np.array([1, 1]),   # Gamba anteriore destra
        np.array([-1, -1]), # Gamba posteriore sinistra
        np.array([1, -1])   # Gamba posteriore destra
    ]
    
    plt.figure()
    
    # Traccia il corpo come un punto all'origine
    plt.plot(0, 0, 'ro', label='Centro del corpo')

    # Traccia ogni gamba come un segmento
    for i, (length, angle) in enumerate(leg_movements):
        start = start_positions[i]
        end = start + np.array([length * np.cos(angle), length * np.sin(angle)])
        plt.plot([start[0], end[0]], [start[1], end[1]], label=f'Gamba {i+1}')
    
    plt.title(f'Movimenti delle Gambe (Segmenti) - Generazione {generation}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

# Algoritmo genetico
def genetic_algorithm(pop_size=50, generations=1400, mutation_rate=0.1, fitness_threshold=10.0):
    population = initialize_population(pop_size)
    best_individual = None
    best_fitness = float('-inf')  # Inizializziamo la migliore fitness al valore minimo possibile
    
    for generation in range(generations):
        fitnesses = [swimming_efficiency(individual) for individual in population]
        current_best_fitness = max(fitnesses)
        current_best_individual = population[fitnesses.index(current_best_fitness)]
        
        # Aggiorna la migliore soluzione trovata se quella attuale è migliore
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
        
        print(f"Generazione {generation}: Miglior Fitness = {best_fitness:.4f}")
        
        # Visualizza i movimenti delle gambe per il miglior individuo della generazione corrente
        plot_leg_movements(current_best_individual, generation)
        
        # Controlla se abbiamo raggiunto la soglia di fitness
        if best_fitness >= fitness_threshold:
            print(f"Obiettivo raggiunto alla generazione {generation} con fitness {best_fitness:.4f}")
            save_solution(best_individual)  # Salva la soluzione migliore
            break
        
        # Genera la prossima generazione
        next_population = []
        while len(next_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_population.append(child)
        
        population = next_population
    
    # Se il ciclo termina senza aver raggiunto la soglia di fitness, salva comunque la migliore soluzione trovata
    if best_fitness < fitness_threshold:
        print(f"Massimo numero di generazioni raggiunto. Miglior fitness: {best_fitness:.4f}")
        save_solution(best_individual)  # Salva la soluzione migliore anche se non ha raggiunto la soglia

    
    print("Evoluzione completata.")

# Esegui l'algoritmo genetico
genetic_algorithm()

