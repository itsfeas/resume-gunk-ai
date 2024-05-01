from collections import defaultdict
from functools import cache
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
from pdf2image import convert_from_path

