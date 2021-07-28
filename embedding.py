import numpy as np
import h5py
import pickle
import argparse
import os

def load_embedding(path='embeddings/embedding_matrix.h5'):
	with h5py.File(path,'r') as hf:
		# data = hf.get('embedding_matrix')
		embedding_matrix = np.array(hf.get('embedding_matrix'))
	return embedding_matrix

def save_embedding(embedding_matrix, path='embeddings/embedding_matrix.h5'):
	with h5py.File(path, 'w') as hf:
		hf.create_dataset('embedding_matrix', data=embedding_matrix)

def load_idx(path='embeddings/word_idx'):
	with open(path,'rb') as file:
		word_idx = pickle.load(file)
	return word_idx

def save_idx(word_idx, path='embeddings/word_idx'):
	with open(path,'wb') as f:
		pickle.dump(word_idx,f)

def create(glove_path, dim, word_idx):

	if not os.path.isfile(glove_path):
		print(f"Cant find file {glove_path}")
		return

	embeddings_dict = {}
	
	with open(glove_path,'r', encoding='utf-8') as f:
		for line in f:
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			assert(coefs.shape == (dim,))
			embeddings_dict[word] = coefs

	embedding_matrix = np.zeros((len(word_idx)+1, dim))

	for w, i in word_idx.items():
		embedding_matrix[i] = embeddings_dict.get(w, np.zeros(dim))
		
	return embedding_matrix


def create_from_top_words(glove_path, dim, num_words=50000):
	'''get top num_words from glove and returns word_idx and embedding_matrix'''

	if not os.path.isfile(glove_path):
		print(f"Cant find file {glove_path}")
		return

	embeddings_dict = {}
	word_idx = {}
	
	with open(glove_path,'r', encoding='utf-8') as f:
		lines = f.readlines()
		for i in range(num_words):
			values = lines[i].split(' ')
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			assert(coefs.shape == (dim,))
			embeddings_dict[word] = coefs
			word_idx[word] = i+1

	embedding_matrix = np.zeros((num_words+1, dim))

	for w, i in word_idx.items():
		embedding_matrix[i] = embeddings_dict[w]
		
	return word_idx, embedding_matrix
