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
from .split_manual import split_resume_to_sections
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')

MODEL_NAME = "resume_parser"
# CLOSENESS_WORDS = ['', 'microservices', 'by', 'experience', 'firebase', 'led', 'automated', 'languages', 'automation', 'tools', 'developed', 'java', 'volume', 'development', 'on', 'developer', 'pipeline', 'tool', 'application', 'nodejs', 'aws', 'android', 'model', 'from', 'program', 'club', 'fullstack', 'technical', 'developing', 'c', 'time', 'over', 'docker', 'frontend', 'python', 'manual', 'integrated', 'project', 'engineering', 'leadership', 'computer', 'api', 'selenium', 'designed', 'web', 'work', 'coop', 'numpy', 'git', 'the', 'sql', 'skills', 'data', 'and', 'to', 'javascript', 'using', 'implemented', 'projects', 'saving', 'sqlite', 'in', 'a', 'typescript', 'under', 'scalable', 'hours', 'analysis', 'software', 'of', 'react', 'junit', 'for', 'university', 'alberta', 'database', 'education', 'PHONE', 'EMAIL', 'NUM', 'MONTH']
dev_skills = ["Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "HTML", "CSS", "SQL", "MongoDB", "Firebase", "PostgreSQL", "MySQL", "GraphQL", "JSON", "XML", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "Bitbucket", "SVN", "Subversion", "Jira", "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "CICD", "Linux", "Unix", "Bash", "Shell", "scripting", "AWS", "Azure", "GCP", "Heroku", "DigitalOcean", "OpenStack", "Virtualization", "Infrastructure", "sklearn", "IaC", "Terraform", "Ansible", "Puppet", "Chef", "Jenkins", "CI", "CircleCI", "Selenium", "Cypress", "JUnit", "NUnit", "TestNG", "Mocha", "Jest", "NUnit", "PyTest", "Maven", "Gradle", "npm", "Yarn", "Webpack", "Babel", "ESLint", "Prettier", "VS", "ECS", "Fargate", "EC2",
"CockroachDB" "IntelliJ", "Eclipse", "Sublime", "Atom", "PyGame", "Vim", "Jupyter", "Sublime", "Atom", "Ionic", "Vim", "Jupyter", "TensorFlow", "PyTorch", "Keras", "OpenCV", "Pandas", "NumPy", "SciPy", "Matplotlib", "D3js", "React", "ReactJs", "Angular", "Rust", "scikitlearn", "Bootstrap" "AngularJs", "Vue", "Vuejs", "Vue", "Redux", "MobX", "Express" "Expressjs", "Django", "Flask", "Ruby",
"Rails" "Laravel", "Spring", "Boot", "NET", "ASPNET", "Nodejs", "GraphQL", "Apollo", "SOAP", "OAuth", "JWT", "OAuth2", "OpenID", "WebSocket", "gRPC", "RabbitMQ", "Kafka", "MQTT", "ActiveMQ", "Elasticsearch", "Logstash", "Kibana", "Splunk", "Grafana", "Prometheus", "Nagios", "JMeter", "Gatling", "SSL/TLS", "OWASP", "Blockchain", "Ethereum", "Solidity", "Corda", "Truffle", "Ganache", "Web3js", "Chrome", "Firefox", "Safari", "iOS", "Android", "Flutter", "Xamarin", "Ionic", "AR", "VR", "Unity", "Unreal", "UXUI", "UX", "UI", "Wireframing", "Prototyping", "Sketch", "Sqlite", "Figma", "InVision", "Zeplin", "Zeppelin", "Confluence", "Slack", "JupyterHub", "Office", "Trello", "Asana", "GitLab", "Git", "nextjs", "prisma", "puppeteer", "Bitbucket", "GitKraken",
"SourceTree", "matlab"
]
CLOSENESS_WORDS = ['', 'linkedin', 'github', 'stopword', 'MONTH', 'languages', 'NUM', 'club', 'education', 'PHONE', 'university', 'tools', 'EMAIL', 'leadership', 'skills', 'experience', 'projects', 'software', 'technical', 'model', 'work', 'languages', 'frameworks', 'libraries', 'relevant', 'coursework', 'achievements', 'portfolio']
SECTIONS_DICT = {
	1: ["personal", "projects"],
	2: ["work", "experience", "employment", "history"],
	3: ["education"],
	4: ["volunteering", "volunteer"],
	5: ["extracurriculars", "extracurricular", "activities"],
	6: ["achievements"],
	7: ["awards"],
	8: ["certifications", "relevant", "courses"],
	9: ["technical", "skills"],
	10: ["additional", "other"]
}
KEYWORD_TO_SECTION_DICT = {}
for key, value_list in SECTIONS_DICT.items():
	for item in value_list:
		KEYWORD_TO_SECTION_DICT[item] = key
SECTION_HEADERS_SET = set()
for v in SECTIONS_DICT.values():
	SECTION_HEADERS_SET |= set(v)
SECTION_HEADERS = set(["personal", "projects", "technical", "work", "experience", "education", "volunteering", "volunteer", "extracurriculars", "extracurricular", "activities", "volunteerextracurriculars", "achievements", "certifications", "technical", "skills"])
POS_TAG_SET = list(nltk.load('help/tagsets/upenn_tagset.pickle').keys())
POS_TAG_SET_TRUE = set(POS_TAG_SET)
DEV_SKILLS_SET = set([skill.lower() for skill in dev_skills])
STOPWORDS = stopwords.words('english')
PRUNED_WORD2VEC_PATH = "./dataset/embeddings/trained/pruned.word2vec.txt"
# def closeness_features(text_split: list[str]):

def convert_months(month_pattern, month_mapping, text):
	return re.sub(month_pattern, lambda match: month_mapping[match.group(0)], text)


def closeness_features(text_split: list[str], closeness_words: list[str]):
	d = {}
	for i in range(len(closeness_words)):
		d[closeness_words[i]] = SortedList()
	for i in range(len(text_split)):
		if text_split[i] in d:
			d[text_split[i]].add(i)
	
	l = [0]*len(text_split)
	for i in range(len(text_split)):
		l[i] = [0]*len(d.keys())
		for j, k in enumerate(d.keys()):
			if not d[k]:
				l[i][j] = -1
			else:
				l[i][j] = (i - d[k][d[k].bisect(i)-1])/len(text_split)
	return l

@cache
def word_to_normalized_histogram(word):
	# Define the bins
	bins = 'abcdefghijklmnopqrstuvwxyz'

	# Initialize the histogram with zeros for each bin
	histogram = {char: 0 for char in bins}

	# Count the occurrences of each character in the word
	for char in word:
		if char in histogram:
			histogram[char] += 1

	# Calculate the total number of characters in the word
	total_chars = len(word) if len(word) > 0 else 1
	# Normalize the histogram values by dividing by the total number of characters
	return {char: count / total_chars for char, count in histogram.items()}

@cache
def punctuation_histogram(word):
	# Define the bins
	bins = ':;,.!?•[]|(){}<>""*^&%$#@~+-=0123456789'

	# Initialize the histogram with zeros for each bin
	histogram = {char: 0 for char in bins}

	# Count the occurrences of each character in the word
	for char in word:
		if char in histogram:
			histogram[char] += 1

	# Calculate the total number of characters in the word
	total_chars = len(word) if len(word) > 0 else 1
	# Normalize the histogram values by dividing by the total number of characters
	return {char: count / total_chars for char, count in histogram.items()}

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
		'sep': 'MONTH',
		'sept': 'MONTH',
		'oct': 'MONTH',
		'nov': 'MONTH',
		'dec': 'MONTH',
		'january': 'MONTH',
		'february': 'MONTH',
		'march': 'MONTH',
		'mar': 'MONTH',
		'april': 'MONTH',
		'apr': 'MONTH',
		'may': 'MONTH',
		'june': 'MONTH',
		'jun': 'MONTH',
		'july': 'MONTH',
		'jul': 'MONTH',
		'august': 'MONTH',
		'september': 'MONTH',
		'october': 'MONTH',
		'november': 'MONTH',
		'december': 'MONTH'
	}
	month_pattern = re.compile(r'\b(?:' + '|'.join(month_mapping.keys()) + r')\b', re.IGNORECASE)
	url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b', re.IGNORECASE)
	email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
	phone_pattern = re.compile(r'\b(?:\(*\d{3}\)*[-.\s]?)?\d{3}[-.\s]?\d{4}\b')
	special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
	ampersand_pattern = re.compile(r'\b&\b')

	filter_s = [s for s in split_s]
	# Replace email addresses with 'EMAIL'
	for i in range(len(filter_s)):
		filter_s[i] = filter_s[i].lower()
		filter_s[i] = 'stopword' if filter_s[i] in STOPWORDS else filter_s[i]
		filter_s[i] = re.sub(url_pattern, 'URL', filter_s[i])
		filter_s[i] = re.sub(email_pattern, 'EMAIL', filter_s[i])
		filter_s[i] = re.sub(phone_pattern, 'PHONE', filter_s[i])
		filter_s[i] = re.sub(ampersand_pattern, 'and', filter_s[i])
		filter_s[i] = re.sub(special_char_pattern, '', filter_s[i])
		filter_s[i] = 'SKILL' if filter_s[i] in DEV_SKILLS_SET else filter_s[i]
		filter_s[i] = convert_months(month_pattern, month_mapping, filter_s[i])
		filter_s[i] = 'NUM' if filter_s[i].isnumeric() else filter_s[i]
	print("filter_s", filter_s)
	return filter_s

def load_word2vec_model(path: str):
	# f = open(path, "r")
	# word2vec_sampled = f.read()
	return gensim.models.KeyedVectors.load_word2vec_format(fname=path, binary=False)

def get_pos_tags(text):
	words = regexp_tokenize(text=text, pattern="\S+")
	pos_tags = pos_tag(words)
	return [p[1] if p[1] in POS_TAG_SET_TRUE else "" for p in pos_tags] + ['NNP']

def normalize_df(df: pd.DataFrame):
	scaler = StandardScaler()
	normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
	return normalized_df

def pos_tag_heatmap(pos_tags: list[str], spread = 5):
	heatmaps = np.zeros((len(pos_tags), len(POS_TAG_SET)), dtype=np.float32)
	d = {}
	for i in range(len(POS_TAG_SET)):
		d[POS_TAG_SET[i]] = i
	for i in range(len(pos_tags)-1):
		for j in range(max(0, i-spread), min(len(pos_tags)-1, i+spread)):
			heatmaps[i][d[pos_tags[j]]] += 1
	return normalize_df(pd.DataFrame(heatmaps, columns=POS_TAG_SET))

def count_punctuation(sentence):
	return sum(1 for char in sentence if char in string.punctuation)

def full_feature_set(resume_path: str, model: gensim.models.KeyedVectors):
		text = extract_text(resume_path, laparams= LAParams(char_margin=200, line_margin=1))
		text_split = parse_to_list(text)
		return generate_features(text, text_split, model)

def generate_features(text: str, text_split, model: gensim.models.KeyedVectors):
		text_split = parse_to_list(text)
		print(text_split)
		filtered = filtered_list(text_split)
		pos_tag_feature = get_pos_tags(text)
		print(text)
		print(filtered)

		features = []
		# f_closeness = closeness_features(filtered, CLOSENESS_WORDS)
		# punctuation_counts = [count_punctuation(w) for w in text_split]
		# is_skill = [w in DEV_SKILLS_SET for w in text_split]
		# f_histograms = [list(word_to_normalized_histogram(w).values()) for w in filtered]
		f_embeddings = [model[w] if model.has_index_for(w) else model["null"] for w in filtered]
		is_header = [1.0 if w in SECTION_HEADERS else 0.0 for w in filtered]
		f_punc_histograms = [list(punctuation_histogram(w).values()) for w in text_split]
		# f_len = [len(w) for w in text_split]
		f_uppercase = [(sum(c.isupper() for c in w)/len(w)) if w else 0 for w in text_split]
		f_pos_tag_closeness = closeness_features(pos_tag_feature, POS_TAG_SET)

		# line context
		lines = [l for l in text.splitlines() if l]
		len_lines = [len(l.split()) for l in lines]
		line_sections = split_resume_to_sections(text, -1, SECTION_HEADERS_SET, KEYWORD_TO_SECTION_DICT)
		num_per_line = [[(len(re.sub(pattern=f'\d+', string=l, repl=""))/len(l)) for _ in l.split()] for l in lines]
		section_for_line = [[tagged_line[0] for _ in tagged_line[1].split()] for tagged_line in line_sections]
		pos_in_line = [[(i/length) for i in range(length)] for length in len_lines]
		len_per_line = [[len(l)]*length for l, length in zip(lines, len_lines)]
		f_pos_in_line = []
		for l in pos_in_line: f_pos_in_line.extend(l)
		f_num_per_line = []
		for l in num_per_line: f_num_per_line.extend(l)
		f_len_per_line = []
		for l in len_per_line: f_len_per_line.extend(l)
		f_section_for_line = []
		for l in section_for_line: f_section_for_line.extend(l)

		# df_histograms = pd.DataFrame(f_histograms).add_prefix('hist_')
		# df_closeness = pd.DataFrame(f_closeness).add_prefix('closeness_')
		df_pos_tag = pd.DataFrame(pos_tag_feature).add_prefix('pos_tag_')
		df_pos_tag_closeness = pd.DataFrame(f_pos_tag_closeness).add_prefix('pos_tag_closeness_')
		df_punc_histograms = pd.DataFrame(f_punc_histograms).add_prefix('punc_hist_')
		# df_len = pd.DataFrame(f_len).add_prefix('len_')
		df_num_per_line = pd.DataFrame(f_num_per_line).add_prefix('num_per_line_')
		df_embeddings = pd.DataFrame(f_embeddings).add_prefix('embeddings_')
		df_upper = pd.DataFrame(f_uppercase).add_prefix('uppercase_')
		df_heatmaps = pos_tag_heatmap(pos_tag_feature).add_prefix('heatmap_')
		df_headers = pd.DataFrame(is_header).add_prefix('headers_')
		df_pos_in_line = pd.DataFrame(f_pos_in_line).add_prefix('pos_line_')
		df_len_per_line = normalize_df(pd.DataFrame(f_len_per_line).add_prefix('line_len_'))
		df_section_for_line = pd.DataFrame(f_section_for_line).add_prefix('line_section_')
		# df_is_skill = pd.DataFrame(is_skill)

		# encode
		enc = OrdinalEncoder()
		enc.fit(pd.DataFrame(POS_TAG_SET).add_prefix('pos_tag_'))
		df_pos_tag = pd.DataFrame(enc.transform(df_pos_tag))

		df = pd.concat([df_embeddings, df_upper, df_num_per_line, df_len_per_line,
			df_pos_tag, df_pos_in_line, df_pos_tag_closeness, df_headers, df_punc_histograms, df_heatmaps, df_section_for_line], axis=1)
		df.fillna(0, inplace=True)
		df.columns = df.columns.astype(str)
		
		# imputer = SimpleImputer(strategy='constant', fill_value=0)
		# df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
		return text, df


def random_sampling_remove_null(df: pd.DataFrame, percentage: float):
	# Randomly sample 80% of rows with the label 'NULL'
	mask = (df['Labels'] == 'NULL')
	return df.drop(df[mask].sample(frac=percentage).index)