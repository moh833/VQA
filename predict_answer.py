import numpy as np
import embedding 
import os
import keras.backend as K
from nltk import word_tokenize
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model, Model
# from tensorflow.keras.applications.vgg16 import VGG16
from models import VGG_16
import argparse


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def extract_image_features(img_path):
	# model = VGG16('weights/vgg16_weights.h5')
	model = VGG_16('weights/vgg16_weights.h5')
	img = image.load_img(img_path,target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)

	features_model = Model(inputs=model.input, outputs=model.layers[-1].input)
	features = features_model([x], training=False)
	## another way
	# last_layer_output = K.function(inputs=model.input, outputs=model.layers[-1].input)
	# with eager_learning_phase_scope(value=0):
		# features = last_layer_output([x])
	return features


def preprocess_question(question, word_idx):
	tokens = word_tokenize(question.lower())
	seq = []
	for token in tokens:
		seq.append(word_idx.get(token,0))
	seq = np.reshape(seq,(1,len(seq)))
	return seq

def generate_answer(img_path, question, dataset, model_name):

	word_idx = embedding.load_idx(f'embeddings/word_idx_{dataset}')
	img_features = extract_image_features(img_path)
	question_sequence = preprocess_question(question, word_idx)

	model_path = f'weights/{model_name}_{dataset}.h5'
	model = load_model(model_path)

	x = [img_features, question_sequence]
	probabilities = model.predict(x)[0]
	
	answers_ids = np.argsort(probabilities)[::-1]
	top_answers_classes = load_top_answers(dataset)

	top_predicted_answers = [top_answers_classes[answers_ids[i]] for i in range(5)]
	
	return top_predicted_answers


def load_top_answers(dataset):
	with open(f'top_answers/{dataset}_top_answers.txt', 'r') as file:
		top_answers = [line.strip() for line in file]
	return top_answers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image_path', type=str, required=True, help='The path of the image')
	parser.add_argument('-q', '--question', type=str, required=True)
	parser.add_argument('-m', '--model_name', type=str, choices=['bow', 'lstm_qi', 'lstm_qi_2'], default='lstm_qi_2')
	parser.add_argument('-d', '--dataset', type=str, default='VQA_1', choices=['COCO-QA', 'VQA_1'], help='The dataset that the model was trained on')
	args = parser.parse_args()

	# image_path = 'test_images/COCO_train2014_000000000650.jpg'
	# question = 'what is in the photo?'
	# model_name = 'lstm_qi'
	# dataset = 'COCO-QA'

	if not os.path.isfile(args.image_path):
		raise ValueError("Image file doesn't exist.")

	top_5_answers = generate_answer(args.image_path, args.question, args.dataset, args.model_name)
	print(f"Top answers: {', '.join(top_5_answers)}")

if __name__ == '__main__':
	main()