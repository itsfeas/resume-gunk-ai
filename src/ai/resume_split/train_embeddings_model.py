import gensim
from nltk.data import find

PATH = "./dataset/embeddings/trained/pruned.word2vec.txt"

f = open(PATH, "r")
word2vec_sampled = f.read("./src/ai/resume_split/dataset/embeddings/trained/pruned.word2vec.txt")
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sampled, binary=False)
# model.save('pruned.resume.word2vec.bin')

print(model)