from collections import defaultdict
from functools import cache
from os import walk
from os.path import join, dirname
from dotenv import load_dotenv
import joblib
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text, LAParams
from sklearn import metrics
from sklearn.calibration import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sortedcontainers import SortedList
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from collections import Counter
from nltk import pos_tag
from ai.resume_split.skills import dev_skills
from ai.resume_split.features import full_feature_set, generate_features, parse_to_list
import pandas as pd
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

MODEL_NAME = "resume_parser"
# CLOSENESS_WORDS = ['', 'microservices', 'by', 'experience', 'firebase', 'led', 'automated', 'languages', 'automation', 'tools', 'developed', 'java', 'volume', 'development', 'on', 'developer', 'pipeline', 'tool', 'application', 'nodejs', 'aws', 'android', 'model', 'from', 'program', 'club', 'fullstack', 'technical', 'developing', 'c', 'time', 'over', 'docker', 'frontend', 'python', 'manual', 'integrated', 'project', 'engineering', 'leadership', 'computer', 'api', 'selenium', 'designed', 'web', 'work', 'coop', 'numpy', 'git', 'the', 'sql', 'skills', 'data', 'and', 'to', 'javascript', 'using', 'implemented', 'projects', 'saving', 'sqlite', 'in', 'a', 'typescript', 'under', 'scalable', 'hours', 'analysis', 'software', 'of', 'react', 'junit', 'for', 'university', 'alberta', 'database', 'education', 'PHONE', 'EMAIL', 'NUM', 'MONTH']
CLOSENESS_WORDS = ['', 'stopword', 'using', 'MONTH', 'fullstack', 'application', 'c', 'languages', 'NUM', 'club', 'api', 'education', 'implemented', 'frontend', 'integrated', 'database', 'PHONE', 'engineering', 'university', 'developed', 'SKILL', 'web', 'tools', 'EMAIL', 'data', 'leadership', 'alberta', 'skills', 'experience', 'developer', 'projects', 'designed', 'software', 'technical', 'model']
POS_TAG_SET = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
DEV_SKILLS_SET = set([skill.lower() for skill in dev_skills])
STOPWORDS = stopwords.words('english')
# def closeness_features(text_split: list[str]):

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

if __name__ == "__main__":
	# Load environment variables from the .env file
	dotenv_path = join(dirname(__file__), 'env/.env')
	load_dotenv(dotenv_path)

	# for page_layout in extract_pages("./src/dataset/resume.pdf"):
	# 	for element in page_layout:
	# 		print(element)
	tot_features = []
	# for i, resume_path in enumerate(PDFS):
	# 	_, df = full_feature_set(resume_path)
	# 	df['Labels'] = LABELS[i]
	# 	# print(df)
	# 	tot_features.append(df)
	PATH = "./src/ai/resume_split/dataset/"
	for i, p in enumerate(list_files(PATH)):
		if not p.startswith("train_"):
			continue
		text = extract_text(PATH+p, laparams=LAParams(char_margin=200, line_margin=1))
		labels = parse_label_file(PATH+"labelled_words/"+replace_extension(p))
		Y, X = split_lists(labels)
		filtered, df = full_feature_set(PATH+p)
		df['Labels'] = Y
		# print(df)
		tot_features.append(df)
	# for f in tot_features:
	# 	print(f.columns)
	df = pd.concat(tot_features, axis=0, ignore_index=True)
	
	# Random Sampling
	df = random_sampling_remove_null(df, 0.75)

	#Split data into a training and testing set
	Y = df['Labels'].values
	X = df.drop(labels=['Labels'], axis=1)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=19)

	#Train the model
	print("Training Model...")
	# model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=[100, 200, 30])
	# model = MLPClassifier(random_state=1, early_stopping=True, max_iter=2500,
	# 				   learning_rate="adaptive", hidden_layer_sizes=[2224, 2341, 2200, 2200, 2200])
	# model = MLPClassifier(random_state=1, early_stopping=True, max_iter=10000, hidden_layer_sizes=[1299, 1102, 1680, 1924, 1508])
	model = RandomForestClassifier(n_estimators = 500, random_state = 42, max_depth=40)
	# model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
	#  intercept_scaling=1, loss='squared_hinge', max_iter=100000,
	#  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
	#  verbose=0)
	# model = SVC(C=1.0, cache_size=1000, class_weight=None, coef0=0.0, degree=3,
	# 	gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
	# 	shrinking=True, tol=0.0001, verbose=False)
	model.fit(X_train, Y_train)

	# Test accuracy on both sets
	print("Testing Accuracy...")
	prediction_test_train = model.predict(X_train)
	prediction_test = model.predict(X_test)

	print ("Accuracy on training data = ", metrics.accuracy_score(Y_train, prediction_test_train))
	print ("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

	# # Rank features by importance
	# importances = list(model.feature_importances_)
	# feature_list = list(X.columns)
	# feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
	# print(feature_imp)

	#Save Model to file
	print("Dumping Model...")
	joblib.dump(model, open(MODEL_NAME, 'wb'))