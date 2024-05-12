from collections import defaultdict
from functools import cache
from os import walk
from os.path import join, dirname
from dotenv import load_dotenv
import joblib
import numpy as np
from pdfminer.high_level import extract_pages, LAParams
from sklearn import metrics
from sklearn.calibration import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import sklearn.cluster as cluster
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from features import iter_lines

MODEL_NAME = "resume_parser_split"
ORDINAL_ENCODER_NAME = "ordinal_encoder_split"

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

def split_lines_with_labels(lines):
    labeled_list = [] 
    for line in lines:
        if line.strip():  # Skip empty lines
            words = line.split()
            label = words[0]
            labeled_list.append(label)
    return labeled_list

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=19)
    # model = xgb.XGBClassifier()
    # model = xgb.XGBClassifier(max_depth=30, n_estimators=1000)
    # model = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=[150]*2)
    # model = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=[107, 412], early_stopping=True)
    model = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=[489, 244, 261], early_stopping=True)
    # model = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=[700, 500])
    # model = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=[50, 50, 2])
    # model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
	#  intercept_scaling=1, loss='squared_hinge',
	#  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
	#  verbose=0)
    # model = SVC(kernel="rbf", degree=3, class_weight="balanced")

    #Train the model
    print(X_train)
    print("Training Model...")

    # model.fit(X_train, Y_train, sample_weight=compute_sample_weight("balanced", Y_train))
    model.fit(X_train, Y_train)

    # Test accuracy on both sets
    print("Testing Accuracy...")
    prediction_test_train = model.predict(X_train)
    prediction_test = model.predict(X_test)

    print("Accuracy on training data = ", metrics.accuracy_score(Y_train, prediction_test_train))
    print("F1 on training data = ", metrics.f1_score(Y_train, prediction_test_train, average="weighted"))
    print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))
    print("F1 = ", metrics.f1_score(Y_test, prediction_test, average="weighted"))

    #Save Model to file
    print("Dumping Model...")
    joblib.dump(model, open(MODEL_NAME, 'wb'))
    joblib.dump(encoder, open(ORDINAL_ENCODER_NAME, 'wb'))

    try:
        # Rank features by importance
        importances = list(model.feature_importances_)
        feature_list = list(X.columns)
        feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
        feature_imp = list(zip(feature_imp.index, feature_imp.values))
        print(feature_imp)
    except:
        pass