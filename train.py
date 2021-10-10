import numpy as np
import pandas as pd
import prepare_data
import models
import sys
import os
import embedding
from keras.utils.vis_utils import plot_model
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=['vis_blstm', 'lstm_qi', 'lstm_qi_2'])
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['COCO-QA', 'VQA_1'], help="The dataset to train on")
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bz', '--batch_size', type=int, default=128)
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dataset = args.dataset

    data_path = f'processed_data/{dataset}'
    glove_path = 'embeddings/glove.6B.300d.txt'
    glove_dim = 300

    train_data = pd.read_pickle(data_path + '/train')
    val_data = pd.read_pickle(data_path + '/val')

    max_answers = 1000
    # get top answers
    top_answers = prepare_data.get_top_answers(max_answers, train_data, val_data)
    atoi = {w:i for i,w in enumerate(top_answers)}
    # itoa = {i+1:w for i,w in enumerate(top_ans)}

    if not os.path.exists('top_answers'):
        os.makedirs('top_answers')
    with open(f'top_answers/{dataset}_top_answers.txt', 'w') as file:
        file.write('\n'.join(top_answers))
    
    # filter question, which isn't in the top answers.
    filtered_train = prepare_data.filter_questions(train_data, atoi)
    filtered_val = prepare_data.filter_questions(val_data, atoi)

    # create new word_idx and embedding_matrix from the words in the data
    word_idx, embedding_matrix = prepare_data.create_vocab(filtered_train, filtered_val, glove_path, glove_dim)
    # word_idx, embedding_matrix = embedding.create_from_top_words(glove_path, glove_dim, 100000)

    # save the word_idx and the embedding_matrix
    if not os.path.exists('embeddings/'):
        os.makedirs('embeddings/')
    embedding.save_idx(word_idx, 'embeddings/' + f'word_idx_{dataset}')
    # embedding.save_embedding(embedding_matrix, 'embeddings/' + f'embedding_matrix_{dataset}')

    ## load a pre-existing word_idx and the embedding_matrix
    # word_idx = embedding.load_idx('embeddings/' + f'word_idx_{dataset}')
    # embedding_matrix = embedding.load_embedding('embeddings/' + f'embedding_matrix_{dataset}')

    print('Loading questions ...')
    # tokenize and preprocess training questions
    questions_train = prepare_data.questions_matrix(filtered_train, word_idx)
    print('questions\' shape', questions_train.shape)
    # tokenize and preprocess testing questions
    questions_val = prepare_data.questions_matrix(filtered_val, word_idx)

    print('Loading answers ...')
    answer_to_onehot_dict = prepare_data.answer_to_onehot(top_answers)
    answers_train = prepare_data.answers_matrix(filtered_train, answer_to_onehot_dict)
    print('answers\' shape', answers_train.shape)

    answers_val = prepare_data.answers_matrix(filtered_val, answer_to_onehot_dict)


    print('Loading image features ...')
    img_features_train = prepare_data.get_coco_features(filtered_train)
    print('image features\' shape', img_features_train.shape)
    img_features_val = prepare_data.get_coco_features(filtered_val)

    print('Creating model ...')
	
    if model_name == 'lstm_qi':
        model = models.lstm_qi(num_classes=len(top_answers), vocab_size=len(word_idx)+1, embedding_matrix=embedding_matrix)
    elif model_name == 'lstm_qi_2':
        model = models.lstm_qi_2(num_classes=len(top_answers), vocab_size=len(word_idx)+1, embedding_matrix=embedding_matrix)
    elif model_name == 'vis_blstm':
        model = models.vis_blstm(num_classes=len(top_answers), vocab_size=len(word_idx)+1, embedding_matrix=embedding_matrix)

    X_train = [img_features_train, questions_train]
    X_val = [img_features_val, questions_val]
    model.summary()

    print("X_train", X_train[0].shape, X_train[1].shape)
    print("y_train", answers_train.shape)
    print("X_val", X_val[0].shape, X_val[1].shape)
    print("y_val", answers_val.shape)

    def print_fn(s):
        with open(f'{model_name}_{dataset}.txt','a') as f:
            print(s, file=f)
    model.summary(print_fn=print_fn)
    plot_model(model, to_file=f'{model_name}_{dataset}.png', show_shapes=True, show_layer_names=True)
    
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(X_train,answers_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_val,answers_val),
        verbose=1)

    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    model.save('weights/' + f'/{model_name}_{dataset}.h5')

if __name__ == '__main__':
    main()
