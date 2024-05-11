from os import walk
import joblib
import numpy as np
from pyeasyga import pyeasyga
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pdfminer.high_level import extract_text, LAParams, extract_pages
from sklearn.preprocessing import OrdinalEncoder
from features import full_feature_set, iter_lines, random_sampling_remove_null
from script import split_lines_with_labels
import pandas as pd

MODEL_NAME = "resume_parser_split"
ORDINAL_ENCODER_NAME = "ordinal_encoder_split"

def list_files(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return(f)

# Function to create and train the neural network
def create_nn(hidden_layers):
    print(hidden_layers)
    for h in hidden_layers:
        if h == 0:
            return 0
    model = MLPClassifier(random_state=1, early_stopping=True, max_iter=5000, hidden_layer_sizes=hidden_layers)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define the fitness function for the genetic algorithm
def fitness(individual, data):
    hidden_layers = individual
    accuracy = create_nn(hidden_layers)
    return accuracy

if __name__ == "__main__":
    pdf_path = "./dataset/"
    label_path = "./dataset/labelled_lines/"
    pdf_files = list_files(pdf_path)
    tot_features = []
    for pdf_file_name in pdf_files:
        if not pdf_file_name.startswith("train_"):
            continue
        print(pdf_file_name)
        label_file_name = label_path+pdf_file_name
        label_file_name = label_file_name.replace(".pdf", ".txt")

        params = LAParams(char_margin=200, line_margin=1)
        pages = extract_pages(pdf_path+pdf_file_name, laparams=params)
        df_local = iter_lines(pages)
        print(df_local)
        labels = []
        with open(label_file_name, "r", encoding="utf8") as file:
            labels = split_lines_with_labels([l for l in file.readlines() if l.strip() != ""])
        # [print(label, l) for label, l in zip(labels, df_local)]
        df_local['Labels'] = labels
        tot_features.append(df_local)
        # img_files = list_files(img_path)[:max_files]
        # for file in img_files:
        # 	convert_img(img_path+file)
    df = pd.concat(tot_features, axis=0, ignore_index=True)

    # remove all non-header classes
    df['Labels'] = df['Labels'].apply(lambda x: '[NULL]' if x != '[HEADER]' else x)
    # Convert labels to numeric class values
    encoder = OrdinalEncoder()
    df['Labels'] = encoder.fit_transform(df[['Labels']])

    #Split data into a training and testing set
    Y = df['Labels'].values
    X = df.drop(labels=['Labels'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=19)
    # model = xgb.XGBClassifier()
    # model = xgb.XGBClassifier(max_depth=30, n_estimators=1000)
    # model = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=[500, 500, 500])
    # model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    #  intercept_scaling=1, loss='squared_hinge',
    #  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #  verbose=0)
    # model = SVC(kernel="rbf", degree=3, class_weight="balanced")

    # Define the genetic algorithm parameters
    population_size = 10
    generations = 7
    mutation_probability = 0.2

    # Create a genetic algorithm instance
    ga = pyeasyga.GeneticAlgorithm(seed_data=None, population_size=population_size, generations=generations,
                        crossover_probability=0.8, mutation_probability=mutation_probability,
                        elitism=True, maximise_fitness=True)

    # Define the population
    ga.create_individual = lambda _: [np.random.randint(400, 2000) for _ in range(np.random.randint(2, 5))]
    ga.fitness_function = fitness

    # Run the genetic algorithm
    ga.run()

    # Get the best individual and its fitness value
    best_individual, best_fitness = ga.best_individual()

    # Save the best_individual and best_fitness in a file
    with open("genetic.txt", "w") as file:
        file.write(f"Best Individual (Hidden Layers, Neurons per Layer): {best_individual}\n")
        file.write(f"Best Fitness: {best_fitness}")


    # Print the results
    print("Best Individual (Hidden Layers, Neurons per Layer):", best_individual)
    print("Best Fitness:", best_fitness)