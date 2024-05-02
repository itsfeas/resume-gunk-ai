from collections import defaultdict
from functools import cache
from os import walk
from os.path import join, dirname
from dotenv import load_dotenv
import string
import joblib
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text, LAParams
import gensim
from sklearn import metrics
from sklearn.calibration import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sortedcontainers import SortedList
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from collections import Counter
from nltk import pos_tag
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2
import visualize_pages
from pdf_mining import collect_headers

def list_files(mypath):
	f = []
	for (dirpath, dirnames, filenames) in walk(mypath):
		f.extend(filenames)
		break
	return(f)

def convert_img(img_path: str):
	img = cv2.imread(img_path)
	cv2.imshow("image", img)
	# img_row_expand = img.copy()
	# for i in range(img_row_expand.shape[0]):
	# 	if np.any(img_row_expand[i] != [255, 255, 255]):
	# 		img_row_expand[i] = [0, 0, 0]
	img_column_expand = img.copy()
	for j in range(img.shape[1]):
		if np.any(img[:, j] != [255, 255, 255]):
			img_column_expand[:, j] = [0, 0, 0]
	cv2.imshow("img_column_expand", img_column_expand)
	
	img_expand = img.copy()
	dil_size = 1
	dil_element = cv2.getStructuringElement(cv2.MORPH_DILATE, (2 * dil_size + 1, 2 * dil_size + 1), (dil_size, dil_size))

	img_expand = cv2.cvtColor(img_expand, cv2.COLOR_BGR2GRAY)
	img_expand = cv2.equalizeHist(img_expand)
	img_expand = cv2.inRange(img_expand, (0), (125))
	img_expand = cv2.dilate(img_expand, dil_element)
	img_expand = cv2.inRange(img_expand, (0), (125))
	cv2.imshow("img_expand", img_expand)
	cv2.waitKey(0)

def get_header_b_boxes(path: str):
	params = LAParams(char_margin=200, line_margin=1)
	pages = extract_pages(path, laparams=params)
	# for page in pages:
		# page.analyze(params)
		# visualize_pages.show_ltitem_hierarchy(page)
	print([l.get_text() for l in collect_headers(pages)])
  		
	
if __name__ == "__main__":
	pdf_path = "./dataset/"
	img_path = "./dataset/img/"
	file_name = "train_resume1.pdf"
	traverse = False
	traverse = True
	max_files = 5
	if traverse:
		pdf_files = list_files(pdf_path)
		for file in pdf_files[:max_files]:
			bboxes = get_header_b_boxes(pdf_path+file)
			# print(bboxes)
		# img_files = list_files(img_path)[:max_files]
		# for file in img_files:
		# 	convert_img(img_path+file)
	else:
		bboxes = get_header_b_boxes(pdf_path+file_name)
		convert_img(img_path+file_name.replace(".pdf", "_0.pdf"))
	cv2.destroyAllWindows()