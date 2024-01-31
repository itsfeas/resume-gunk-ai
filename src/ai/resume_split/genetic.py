from os import walk
import numpy as np
from pyeasyga import pyeasyga
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pdfminer.high_level import extract_text, LAParams
from features import full_feature_set, random_sampling_remove_null
import pandas as pd

def split_lists(labeled_list):
    labels = [label for label, _ in labeled_list]
    words = [word for _, word in labeled_list]
    return labels, words

def replace_extension(file_name):
	# Find the position of the last dot in the file name
	last_dot_index = file_name.rfind('.')
	
	if last_dot_index != -1:
		# Extract the file name without the extension
		base_name = file_name[:last_dot_index]
		
		# Replace the extension with ".txt"
		new_file_name = base_name + '.txt'
		
		return new_file_name
	else:
		# If there is no dot in the file name, add ".txt" directly
		return file_name + '.txt'

def parse_label_file(file_path):
	data = []
	with open(file_path, 'r', encoding="utf8") as file:
		for line in file:
			# Split each line into two parts using whitespace as a separator
			parts = line.strip().split()
			
			# Ensure there are exactly two parts (LABEL and VAL)
			if len(parts) == 2:
				label, value = parts
				# Convert the value to the appropriate data type (e.g., int, float)
				# based on your specific needs
				data.append((label, value))
			else:
				# Handle cases where the line doesn't have the expected format
				print(f"Skipping line: {line.strip()}")

	return data

def list_files(mypath):
	f = []
	for (dirpath, dirnames, filenames) in walk(mypath):
		f.extend(filenames)
		break
	return(f)

def random_sampling_remove_null(df: pd.DataFrame, percentage: float):
	# Randomly sample 80% of rows with the label 'NULL'
	mask = (df['Labels'] == 'NULL')
	return df.drop(df[mask].sample(frac=percentage).index)

tot_features = []
PATH = "./dataset/"
print(list_files(PATH))
for i, p in enumerate(list_files(PATH)):
    if not p.startswith("train_"):
        continue
    text = extract_text(PATH+p, laparams=LAParams(char_margin=200))
    labels = parse_label_file(PATH+"labelled_words/"+replace_extension(p))
    Y, X = split_lists(labels)
    filtered, df = full_feature_set(PATH+p)
    df['Labels'] = Y
    tot_features.append(df)
df = pd.concat(tot_features, axis=0, ignore_index=True)

# Random Sampling
df = random_sampling_remove_null(df, 0.75)

#Split data into a training and testing set
Y = df['Labels'].values
X = df.drop(labels=['Labels'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=19)

# Function to create and train the neural network
def create_nn(hidden_layers):
    print(hidden_layers)
    for h in hidden_layers:
        if h == 0:
            return 0
    model = MLPClassifier(random_state=1, early_stopping=True, max_iter=10000, hidden_layer_sizes=hidden_layers)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define the fitness function for the genetic algorithm
def fitness(individual, data):
    hidden_layers = individual
    accuracy = create_nn(hidden_layers)
    return accuracy

# Define the genetic algorithm parameters
population_size = 10
generations = 20
mutation_probability = 0.2

# Create a genetic algorithm instance
ga = pyeasyga.GeneticAlgorithm(seed_data=None, population_size=population_size, generations=generations,
                      crossover_probability=0.8, mutation_probability=mutation_probability,
                      elitism=True, maximise_fitness=True)

# Define the population
ga.create_individual = lambda _: [np.random.randint(400, 2500) for _ in range(np.random.randint(2, 8))]
ga.fitness_function = fitness

# Run the genetic algorithm
ga.run()

# Get the best individual and its fitness value
best_individual, best_fitness = ga.best_individual()

# Save the best_individual and best_fitness in a file
with open("genetic.txt", "w") as file:
    file.write("Best Individual (Hidden Layers, Neurons per Layer): {}\n".format(best_individual))
    file.write("Best Fitness: {}".format(best_fitness))


# Print the results
print("Best Individual (Hidden Layers, Neurons per Layer):", best_individual)
print("Best Fitness:", best_fitness)