import numpy as np
import h5py
import pickle
import argparse

def load():
	path = 'embeddings/embedding_matrix.h5'
	with h5py.File(path,'r') as hf:
		# data = hf.get('embedding_matrix')
		embedding_matrix = np.array(hf.get('embedding_matrix'))
	return embedding_matrix

def load_idx():
	path = 'embeddings/word_idx'
	with open(path,'rb') as file:
		word_idx = pickle.load(file)
	return word_idx

def create(glove_path, dim):
	embedding_matrix_path = 'embeddings/embedding_matrix.h5'
	word_idx_path = 'embeddings/word_idx'
	embeddings = {}
	word_idx = {}
	
	with open(glove_path,'r', encoding='utf-8') as f:
		for i, line in enumerate(f):
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			assert(coefs.shape == (dim,))
			embeddings[word] = coefs
			word_idx[word] = i+1

	num_words = len(word_idx)
	embedding_matrix = np.zeros((1+num_words,dim))

	for i, word in enumerate(word_idx.keys()):
		embedding_matrix[i+1] = embeddings[word]
		
	with h5py.File(embedding_matrix_path, 'w') as hf:
		hf.create_dataset('embedding_matrix',data=embedding_matrix)

	with open(word_idx_path,'wb') as f:
		pickle.dump(word_idx,f)

def main():
	print('Preparing embeddings ...')
	glove_path = 'embeddings/glove.6B.300d.txt'
	create(glove_path, 300)


if __name__ == '__main__':
	main()