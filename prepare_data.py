import numpy as np
import pandas as pd
from collections import defaultdict
import operator
import sys
from nltk import word_tokenize
import embedding as ebd
from keras.preprocessing.sequence import pad_sequences
from scipy import io
import os

# dataset = ''

# def set_dataset(dataset):
# 	global dataset
# 	dataset = dataset_name

def get_top_answers(dataset):
    data_path = f'processed_data/{dataset}/train'
    df = pd.read_pickle(data_path)
    # list of lists where each list contains the single answer
    answers = df[['answer']].values.tolist()
    
    freq = defaultdict(int)
    for answer in answers:
        # get the lower answer as a key and increase the occurrences by 1
        freq[answer[0].lower()] += 1

    # sort from the most occurrences -list of (answer, occurrences) pairs-
    top_answers_occ = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)[0:1000]
    # list of answers sorted
    top_answers = [answer_occ[0] for answer_occ in top_answers_occ]
    return top_answers


def save_top_answers(dataset):
	top_answers = get_top_answers(dataset)
	if not os.path.exists('top_answers'):
		os.makedirs('top_answers')
	with open(f'top_answers/{dataset}_top_answers.txt', 'w') as file:
		file.write('\n'.join(top_answers))

def load_top_answers(dataset):
	with open(f'top_answers/{dataset}_top_answers.txt', 'r') as file:
		top_answers = [line.strip() for line in file]
	return top_answers


def answers_to_onehot(dataset):
	# new
	save_top_answers(dataset)
	top_answers = get_top_answers(dataset)
    # dictionary of (answer, onehot) values
	answer_to_onehot = {}
	for i, answer in enumerate(top_answers):
		onehot = np.zeros(1001)
		onehot[i] = 1.0
		answer_to_onehot[answer] = onehot
	return answer_to_onehot

# answer_to_onehot_dict = answers_to_onehot()

def get_answers_matrix(dataset, split):
	answer_to_onehot_dict = answers_to_onehot(dataset)
	try:
		data_path = f'processed_data/{dataset}/{split}'
	except OSError as e:
		print('Invalid split!')
		sys.exit()
	
	df = pd.read_pickle(data_path)
	# list of lists where each list contains the single answer
	answers = df[['answer']].values.tolist()
	# (data_size, 1001)
	answer_matrix = np.zeros((len(answers), 1001))

	default_onehot = np.zeros(1001)
	default_onehot[1000] = 1.0
	
	for i, answer in enumerate(answers):
		answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(), default_onehot)
	
	return answer_matrix


def get_questions_matrix(dataset, split):
	try:
		data_path = f'processed_data/{dataset}/{split}'
	except OSError as e:
		print('Invalid split!')
		sys.exit()
	
	df = pd.read_pickle(data_path)
	questions = df[['question']].values.tolist()
	word_idx = ebd.load_idx()
	seq_list = []
	
	for question in questions:
		# list of words
		words = word_tokenize(question[0].lower())
		seq = []
		for word in words:
			# list of indecies of the words - 0 for unknown
			seq.append(word_idx.get(word,0))
		# list of lists (each one has the indeces of the words in a question)
		seq_list.append(seq)
	# padding zeros in the beginning of each sequence so they ending up with the same length and stack them as a matrix
	question_matrix = pad_sequences(seq_list)
	
	# (248349, 25) (questions length, max question length)
	return question_matrix


def get_coco_features(dataset, split):
	try:
		data_path = f'processed_data/{dataset}/{split}'
	except OSError as e:
		print('Invalid split!')
		sys.exit()

	id_map_path = 'coco_features/coco_vgg_IDMap.txt'
	features_path = 'coco_features/vgg_feats.mat'

	# image ids
	img_labels = pd.read_pickle(data_path)[['image_id']].values.tolist()
	# map image id to index in the matrix
	img_ids = open(id_map_path).read().splitlines()
	# load a (4096, 123287) matlab matrix (features, images) in ['feats']
	features_struct = io.loadmat(features_path)  	

	# image id to index in the matrix
	id_map = {}
	for ids in img_ids:
		ids_split = ids.split()
		id_map[int(ids_split[0])] = int(ids_split[1])

	VGGfeatures = features_struct['feats']
	# 4096
	nb_dimensions = VGGfeatures.shape[0]
	# length of the data
	nb_images = len(img_labels)
	# (length of the data, 4096)
	image_matrix = np.zeros((nb_images,nb_dimensions))

	for i in range(nb_images):
		image_matrix[i,:] = VGGfeatures[:,id_map[ img_labels[i][0] ]]

	return image_matrix


if __name__ == '__main__':
	question_matrix = get_questions_matrix('train')
	print(question_matrix.shape)

