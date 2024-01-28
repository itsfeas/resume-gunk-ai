import numpy as np
from pyeasyga import pyeasyga
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import full_feature_set, random_sampling_remove_null
import pandas as pd
from traindata import PDFS, LABELS

tot_features = []
for i, resume_path in enumerate(PDFS):
    _, df = full_feature_set(resume_path)
    df['Labels'] = LABELS[i]
    # print(df)
    tot_features.append(df)
# for f in tot_features:
# 	print(f.columns)
df = pd.concat(tot_features, axis=0, ignore_index=True)

# Random Sampling
df = random_sampling_remove_null(df, 0.7)

#Split data into a training and testing set
Y = df['Labels'].values
X = df.drop(labels=['Labels'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=19)

# Function to create and train the neural network
def create_nn(hidden_layers, neurons_per_layer):
    print(hidden_layers, neurons_per_layer)
    if hidden_layers == 0 or neurons_per_layer == 0:
        return 0
    model = MLPClassifier(random_state=1, early_stopping=True, max_iter=2500, hidden_layer_sizes=(neurons_per_layer,) * hidden_layers)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define the fitness function for the genetic algorithm
def fitness(individual, data):
    hidden_layers, neurons_per_layer = individual
    accuracy = create_nn(hidden_layers, neurons_per_layer)
    return accuracy

# Define the genetic algorithm parameters
population_size = 10
generations = 5
mutation_probability = 0.2

# Create a genetic algorithm instance
ga = pyeasyga.GeneticAlgorithm(seed_data=None, population_size=population_size, generations=generations,
                      crossover_probability=0.8, mutation_probability=mutation_probability,
                      elitism=True, maximise_fitness=True)

# Define the population
ga.create_individual = lambda _: [np.random.randint(400, 1500) for _ in range(np.random.randint(2, 5))]
ga.fitness_function = fitness

# Run the genetic algorithm
ga.run()

# Get the best individual and its fitness value
best_individual, best_fitness = ga.best_individual()

# Print the results
print("Best Individual (Hidden Layers, Neurons per Layer):", best_individual)
print("Best Fitness:", best_fitness)