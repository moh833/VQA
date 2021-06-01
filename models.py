import numpy as np
import embedding
import prepare_data
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, concatenate, Dropout, Input, Multiply


def lstm_qi(num_classes):
	embedding_matrix = embedding.load()
	embedding_model = Sequential()

	embedding_model.add(Embedding(
		input_dim=embedding_matrix.shape[0],
		output_dim=embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))
	lstm1, state_h, state_c = LSTM(512, return_state=True)(embedding_model.output)
	lang_model = concatenate([state_c, state_h])

	image_model = Sequential()
	image_model.add(Dense(
		units=1024,
		input_dim=4096,
		activation='tanh'))

	out = Multiply()([lang_model, image_model.output])
	out = Dropout(0.5)(out)
	out = Dense(1000, activation='tanh')(out)
	out = Dropout(0.5)(out)
	out = Dense(num_classes,activation='softmax')(out)

	main_model = Model([image_model.input, embedding_model.input], out)

	return main_model
