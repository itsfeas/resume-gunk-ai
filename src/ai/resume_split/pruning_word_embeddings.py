import csv
from pdfminer.high_level import extract_text, LAParams
from os import walk
import gensim
from gensim.models.word2vec import Word2Vec
import re

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

def list_files(mypath):
	f = []
	for (dirpath, dirnames, filenames) in walk(mypath):
		f.extend(filenames)
		break
	return(f)

def filtered_list(split_s: list[str]):
	special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')

	filter_s = [s for s in split_s]
	# Replace email addresses with 'EMAIL'
	for i in range(len(filter_s)):
		filter_s[i] = filter_s[i].lower()
		filter_s[i] = re.sub(special_char_pattern, '', filter_s[i])
	return filter_s

# Load the binary model
model = gensim.models.KeyedVectors.load_word2vec_format('./dataset/embeddings/pretrained/GoogleNews-vectors-negative300.bin.gz', binary = True)

# Only output words that appear in the resume dataset
words = set()
f = csv.reader(open('./dataset/embeddings/data/resume.csv', encoding='utf-8'))
for row in f:
	resume_words = set(filtered_list(row[1].split()))
	words = words.union(resume_words)

PATH = "./dataset/"
for i, p in enumerate(list_files(PATH)):
	if not p.startswith("train_"):
		continue
	text = extract_text(PATH+p, laparams=LAParams(char_margin=200, line_margin=1))
	filtered = set(filtered_list(text.split()))
	words = words.union(filtered)
print(words)



# Output presented word to a temporary file
out_file = './dataset/embeddings/trained/pruned.word2vec.txt'
f = open(out_file,'w', encoding='utf-8')

word_presented = words.intersection(set(model.index_to_key))
f.write('{} {}\n'.format(len(word_presented),len(model['word'])))

for word in word_presented:
    f.write('{} {}\n'.format(word, ' '.join(str(value) for value in model[word])))

f.close()