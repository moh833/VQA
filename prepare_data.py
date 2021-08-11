import numpy as np
import pandas as pd
from scipy import io
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import embedding
import os


def get_top_answers(num_answers, train, val=np.array([])):
	counts = {}
	answers = train['answer'].values.tolist()
	if val.size != 0:
		answers += val['answer'].values.tolist()
	for ans in answers:
		counts[ans.lower()] = counts.get(ans, 0) + 1
	cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
	print('top answer and their counts:')    
	print('\n'.join(map(str,cw[:10])))
    
	top_answers = [c[1] for c in cw[:num_answers]]

	return top_answers


def filter_questions(data, atoi):
    answers = data['answer'].values.tolist()
    indices = []
    for i, ans in enumerate(answers):
        if atoi.get(ans, len(atoi)+1) != len(atoi)+1:
            indices.append(i)

    new_data = data.iloc[indices]
    print('questions number reduced from %d to %d '%(len(data.index), len(new_data.index)))
    return new_data


def create_vocab(train, val=np.array([]), glove_path=None, dim=None, threshold=0):
	counts = {}
	questions = train['question'].values.tolist()
	if val.size != 0:
		questions += val['question'].values.tolist()
	for q in questions:
		words = word_tokenize(q.lower())
		for w in words:
			counts[w] = counts.get(w, 0) + 1
	cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    
	vocab = [c[1] for c in cw if c[0] > threshold]
	print(f'vocab = {len(vocab)} words')
	print('top words in questions and their counts:')    
	print('\n'.join(map(str,cw[:10])))

	word_idx = {w:i+1 for i, w in enumerate(vocab)}

	embedding_matrix = np.array([])
	if glove_path and dim:
		if os.path.isfile(glove_path):
			embedding_matrix = embedding.create(glove_path, dim, word_idx)

	return word_idx, embedding_matrix

def questions_matrix(data, word_idx):
	
	questions = data['question'].values.tolist()
	seq_list = []
	
	for q in questions:
		# list of words
		words = word_tokenize(q.lower())
		seq = []
		for word in words:
			# list of indices of the words - 0 for unknown
			seq.append(word_idx.get(word, 0))
		# list of lists (each one has the indices of the words in a question)
		seq_list.append(seq)
	# padding zeros in the beginning of each sequence so they ending up with the same length and stack them as a matrix
	question_matrix = pad_sequences(seq_list)
	
	# (questions length, max question length)
	return question_matrix


def answer_to_onehot(top_answers):
    # dictionary of (answer, onehot) values
	answer_to_onehot = {}
	num_answers = len(top_answers)
	for i, ans in enumerate(top_answers):
		onehot = np.zeros(num_answers)
		onehot[i] = 1.0
		answer_to_onehot[ans] = onehot
	return answer_to_onehot


def answers_matrix(data, answer_to_onehot_dict):
	
	# list of lists where each list contains the single answer
	answers = data['answer'].values.tolist()
	# (data_size, num_answers)
	answers_matrix = np.zeros((len(answers), len(answer_to_onehot_dict)))
	
	for i, ans in enumerate(answers):
		answers_matrix[i] = answer_to_onehot_dict[ans.lower()]
	
	return answers_matrix

def get_coco_features(data):

	id_map_path = 'coco_features/coco_vgg_IDMap.txt'
	features_path = 'coco_features/vgg_feats.mat'

	# image ids
	img_labels = data[['image_id']].values.tolist()
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



def main():

	dataset = 'COCO-QA'
	data_path = f'processed_data/{dataset}'
	glove_path = 'embeddings/glove.6B.300d.txt'
	glove_dim = 300

	train_data = pd.read_pickle(data_path + '/train')
	val_data = pd.read_pickle(data_path + '/val')

	num_answers = 1000
	# get top answers
	top_answers = get_top_answers(num_answers, train_data, val_data)
	atoi = {w:i for i,w in enumerate(top_answers)}
	# itoa = {i+1:w for i,w in enumerate(top_answers)}

	if not os.path.exists('top_answers'):
		os.makedirs('top_answers')
	with open(f'top_answers/{dataset}_top_answers.txt', 'w') as file:
		file.write('\n'.join(top_answers))

	# filter question, which isn't in the top answers.
	filtered_train = filter_questions(train_data, atoi)
	filtered_val = filter_questions(val_data, atoi)

	word_idx, embedding_matrix = create_vocab(filtered_train, filtered_val, glove_path, glove_dim)

	# tokenize and preprocess training questions
	questions_train = questions_matrix(filtered_train, word_idx)
	# tokenize and preprocess testing questions
	questions_val = questions_matrix(filtered_val, word_idx)


	answer_to_onehot_dict = answer_to_onehot(top_answers)
	answers_train = answers_matrix(filtered_train, answer_to_onehot_dict)
	answers_val = answers_matrix(filtered_val, answer_to_onehot_dict)


if __name__ == "__main__":
    main()