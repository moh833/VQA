import numpy as np
import embedding
import prepare_data
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, concatenate, Reshape, Dropout, Conv2D, MaxPool2D, Flatten, Input, Multiply, Bidirectional, BatchNormalization


def lstm_qi(num_classes, vocab_size, embedding_matrix=np.array([]), embedding_trainable=False):

	embedding_in = Input(shape=(None,))

	if embedding_matrix.size != 0:
		print('Loaded pre-trained embedding')
		embedding_out = Embedding(
			input_dim=embedding_matrix.shape[0],
			output_dim=embedding_matrix.shape[1],
			weights = [embedding_matrix],
			trainable = embedding_trainable)(embedding_in)
	else:	
		embedding_out = Embedding(
			input_dim=vocab_size,
			output_dim=300)(embedding_in)

	lstm1, state_h, state_c = LSTM(512, return_state=True)(embedding_out)
	lang_model = concatenate([state_c, state_h])

	image_in = Input(shape=(4096,))
	image_model = Dense(
		units=1024,
		input_dim=4096,
		activation='tanh')(image_in)

	out = Multiply()([lang_model, image_model])
	out = Dropout(0.5)(out)
	out = Dense(1000, activation='tanh')(out)
	out = Dropout(0.5)(out)
	out = Dense(num_classes,activation='softmax')(out)

	main_model = Model([image_in, embedding_in], out)

	return main_model



def lstm_qi_2(num_classes, vocab_size, embedding_matrix=np.array([]), embedding_trainable=False):
	embedding_in = Input(shape=(None,))

	if embedding_matrix.size != 0:
		print('Loaded pre-trained embedding')
		embedding_out = Embedding(
			input_dim=embedding_matrix.shape[0],
			output_dim=embedding_matrix.shape[1],
			weights = [embedding_matrix],
			trainable = embedding_trainable)(embedding_in)
	else:	
		embedding_out = Embedding(
			input_dim=vocab_size,
			output_dim=300)(embedding_in)

	lstm1, state_h_1, state_c_1 = LSTM(512, return_state=True, return_sequences=True)(embedding_out)
	lstm2, state_h_2, state_c_2 = LSTM(512, return_state=True)(lstm1)
	lang_model = concatenate([state_c_1, state_h_1, state_c_2, state_h_2])
	lang_model = Dense(1024, activation='tanh')(lang_model)

	image_in = Input(shape=(4096,))
	image_norm = BatchNormalization()(image_in)
	image_model = Dense(
		units=1024,
		input_dim=4096,
		activation='tanh')(image_norm)

	out = Multiply()([lang_model, image_model])
	out = Dropout(0.5)(out)
	out = Dense(1000, activation='tanh')(out)
	out = Dropout(0.5)(out)
	out = Dense(num_classes,activation='softmax')(out)

	main_model = Model([image_in, embedding_in], out)

	return main_model


def bow_model(num_classes, vocab_size):
	lang_model = Sequential()
	lang_model.add(Dense(
	units=1024,
	input_dim=vocab_size,
	activation='linear'))

	image_model = Sequential()
	image_model.add(Dense(
	units=1024,
	input_dim=4096,
	activation='linear'))

	x=concatenate(
	[image_model.output,lang_model.output]
	)
	x=Dropout(0.5)(x)
	x = Dense(2048,activation="relu")(x)
	x = Dense(1024,activation='relu')(x)
	x=Dense(num_classes,activation='softmax')(x)
	main_model=Model([image_model.input,lang_model.input],x)

	return main_model


def vis_blstm(num_classes, vocab_size, embedding_matrix=np.array([]), embedding_trainable=False):
	embedding_in = Input(shape=(None,))
	embedding_dim = 300

	if embedding_matrix.size != 0:
		print('Loaded pre-trained embedding')
		embedding_dim = embedding_matrix.shape[1]
		embedding_out = Embedding(
			input_dim=embedding_matrix.shape[0],
			output_dim=embedding_matrix.shape[1],
			weights = [embedding_matrix],
			trainable = embedding_trainable)(embedding_in)
	else:	
		embedding_out = Embedding(
			input_dim=vocab_size,
			output_dim=embedding_dim)(embedding_in)


	image_model = Sequential()
	image_model.add(Dense(
		units=embedding_dim,
		input_dim=4096,
		activation='linear'))
	image_model.add(Reshape((1,embedding_dim)))

	x=concatenate(
	[image_model.output,embedding_out], axis=1	
	)
	x=Bidirectional(LSTM(500))(x)
	x=Dropout(0.5)(x)
	x=Dense(num_classes,activation='softmax')(x)
	main_model=Model([image_model.input,embedding_in],x)

	return main_model


def VGG_16(weights_path=None):
	model = Sequential()
	model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=1000, activation="softmax"))

	if weights_path:
		model.load_weights(weights_path)
	
	return model