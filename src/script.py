from collections import defaultdict
from os.path import join, dirname
from dotenv import load_dotenv
import joblib
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text
from sklearn import metrics
from sklearn.calibration import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sortedcontainers import SortedList
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import nltk
from nltk.tokenize import regexp_tokenize
from collections import Counter
from nltk import pos_tag
from dataset.model.traindata import X, Y
import pandas as pd
import re

MODEL_NAME = "resume_parser"
# def closeness_features(text_split: list[str]):

def convert_months(month_pattern, month_mapping, text):
    return re.sub(month_pattern, lambda match: month_mapping[match.group(0)], text)


def closeness_features(text_split: list[str]):
	d = defaultdict(SortedList)
	for i in range(len(text_split)):
		d[text_split[i]].add(i)
	
	l = [0]*len(text_split)
	for i in range(len(text_split)):
		l[i] = [0]*len(d.keys())
		for j, k in enumerate(d.keys()):
			l[i][j] = d[k][d[k].bisect(i)-1]/len(text_split)
	return l

def word_to_char_histogram(word):
    # Count the occurrences of each character in the word
    char_counts = Counter(word)

    # Total number of characters in the word
    total_chars = sum(char_counts.values())

    # Calculate normalized histogram
    char_histogram = {char: count / total_chars for char, count in char_counts.items()}

    return char_histogram

def parse_to_list(s: str):
	return re.split(r"\s+", s)

def filtered_list(split_s: list[str]):
	month_mapping = {
		'jan': 'MONTH',
		'feb': 'MONTH',
		'march': 'MONTH',
		'april': 'MONTH',
		'may': 'MONTH',
		'june': 'MONTH',
		'july': 'MONTH',
		'aug': 'MONTH',
		'sept': 'MONTH',
		'oct': 'MONTH',
		'nov': 'MONTH',
		'dec': 'MONTH',
		'january': 'MONTH',
		'february': 'MONTH',
		'march': 'MONTH',
		'april': 'MONTH',
		'may': 'MONTH',
		'june': 'MONTH',
		'july': 'MONTH',
		'august': 'MONTH',
		'september': 'MONTH',
		'october': 'MONTH',
		'november': 'MONTH',
		'december': 'MONTH'
	}
	month_pattern = re.compile(r'\b(?:' + '|'.join(month_mapping.keys()) + r')\b', re.IGNORECASE)
	url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b', re.IGNORECASE)
	email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
	special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
	filter_s = [s for s in split_s]
    # Replace email addresses with 'EMAIL'
	for i in range(len(filter_s)):
		filter_s[i] = re.sub(url_pattern, 'URL', filter_s[i])
		filter_s[i] = filter_s[i].lower()
		filter_s[i] = re.sub(special_char_pattern, '', filter_s[i])
		filter_s[i] = convert_months(month_pattern, month_mapping, filter_s[i])
		filter_s[i] = re.sub(email_pattern, 'EMAIL', filter_s[i])
		filter_s[i] = 'NUM' if filter_s[i].isnumeric() else filter_s[i]

	return filter_s

def get_pos_tags(text):
	words = regexp_tokenize(text=text, pattern="\S+")
	pos_tags = pos_tag(words)
	return [p[1] for p in pos_tags] + ['NNP']

if __name__ == "__main__":
	# Load environment variables from the .env file
	dotenv_path = join(dirname(__file__), 'env/.env')
	load_dotenv(dotenv_path)

	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')

	# for page_layout in extract_pages("./src/dataset/resume.pdf"):
	# 	for element in page_layout:
	# 		print(element)
	
	text = extract_text("./src/dataset/resume1.pdf")
	text_split = parse_to_list(text)
	print(text_split)
	filtered = filtered_list(text_split)
	pos_tag_feature = get_pos_tags(text)
	print(text)
	print(filtered)

	features = []
	closeness = closeness_features(filtered)
	pos_tag_closeness = closeness_features(pos_tag_feature)

	df_closeness = pd.DataFrame(closeness, columns=range(len(closeness[0])))
	df_pos_tag = pd.DataFrame(pos_tag_feature, columns=range(len(pos_tag_feature[0][0])))
	df_pos_tag_closeness = pd.DataFrame(pos_tag_closeness, columns=range(len(pos_tag_closeness[0])))


	# encode
	enc = OrdinalEncoder()
	enc.fit(df_pos_tag)
	df_pos_tag = pd.DataFrame(enc.transform(df_pos_tag))

	# df = pd.DataFrame()
	# for i in range(len(features)):
	# 	df = df.append(features[i])
	# print(l)
	df = pd.concat([df_closeness, df_pos_tag, df_pos_tag_closeness], axis=1)
	df['Labels'] = Y
	print(df)
	
	#Split data into a training and testing set
	Y = df['Labels'].values
	X = df.drop(labels=['Labels'], axis=1)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=19)

	#Train the model
	print("Training Model...")
	# model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=[100, 200, 30])
	# model = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=[1000, 2000, 300])
	# model = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth=30)
	model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
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