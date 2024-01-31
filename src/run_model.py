from collections import defaultdict
from functools import cache
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
import pandas as pd
import re
from ai.resume_split.skills import dev_skills
from ai.resume_split.features import full_feature_set, parse_to_list, filtered_list

MODEL_NAME = "resume_parser"
POS_TAG_SET = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]


if __name__ == "__main__":
	# Load environment variables from the .env file
	dotenv_path = join(dirname(__file__), 'env/.env')
	load_dotenv(dotenv_path)

	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')

	text, df = full_feature_set("./src/ai/resume_split/dataset/resume.pdf")
	
	print("Loading Model...")
	loaded_model = joblib.load(open(MODEL_NAME, 'rb'))
	
	print("Predicting...")
	prediction = loaded_model.predict(df)
	
	print(df)
	print(text)
	# print(prediction)

	for pred, x in zip(prediction, parse_to_list(text)):
		print(pred, x)