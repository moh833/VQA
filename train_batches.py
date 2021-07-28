import numpy as np
import pandas as pd
import prepare_data
import models
import sys
import os
import math
import embedding
import glob
from keras.utils.vis_utils import plot_model



def save_batches(X_1, X_2, Y, batchsize, split):

    m = X_1.shape[0]
    num_batches = math.ceil(m/batchsize)

    if os.path.exists(f'batches/{split}'):
        files = glob.glob(f'batches/{split}/*')
        for f in files:
            os.remove(f)
    else:
        os.makedirs(f'batches/{split}')

    for k in range(num_batches):
            batch_X_1 = X_1[k*batchsize : (k+1)*batchsize, :]
            batch_X_2 = X_2[k*batchsize : (k+1)*batchsize, :]
            batch_Y = Y[k*batchsize : (k+1)*batchsize, :]
            np.save(f'batches/{split}/x_1_{k}.npy', batch_X_1)
            np.save(f'batches/{split}/x_2_{k}.npy', batch_X_2)
            np.save(f'batches/{split}/y_{k}.npy', batch_Y)
    return m, num_batches

def iterate_minibatches(minibatchsize, split, num_saved_batches, shuffle=True):
    while True:
        if shuffle:
            saved_batches = np.random.permutation(num_saved_batches)
        else:
            saved_batches = range(num_saved_batches)
        for k in saved_batches:
                batch_X_1 = np.load(f'batches/{split}/x_1_{k}.npy')
                batch_X_2 = np.load(f'batches/{split}/x_2_{k}.npy')
                batch_Y = np.load(f'batches/{split}/y_{k}.npy')

                m = batch_X_1.shape[0]
                num_mini_batches = math.ceil(m/minibatchsize)
                if shuffle:
                    permutation = np.random.permutation(m)
                    batch_X_1 = batch_X_1[permutation, :]
                    batch_X_2 = batch_X_2[permutation, :]
                    batch_Y = batch_Y[permutation, :]
                for i in range(num_mini_batches):
                    mini_batch_X_1 = batch_X_1[i*minibatchsize : (i+1)*minibatchsize, :]
                    mini_batch_X_2 = batch_X_2[i*minibatchsize : (i+1)*minibatchsize, :]
                    mini_batch_Y = batch_Y[i*minibatchsize : (i+1)*minibatchsize, :]
                    yield ([mini_batch_X_1, mini_batch_X_2], mini_batch_Y)
                
def main():

    num_epochs = 10 
    batch_size = 128 
    saved_batches_size = 20480 # 5120
    model_name = 'lstm_qi_2'
    dataset = 'VQA_1'

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

    word_idx = prepare_data.create_vocab(filtered_train, filtered_val, glove_path, glove_dim)

    print('Loading questions ...')
    # tokenization and preprocessing training question
    questions_train = prepare_data.questions_matrix(filtered_train, word_idx)
    print('questions\' shape', questions_train.shape)

    print('Loading answers ...')
    answer_to_onehot_dict = prepare_data.answer_to_onehot(top_answers)
    answers_train = prepare_data.answers_matrix(filtered_train, answer_to_onehot_dict)
    print('answers\' shape', answers_train.shape)


    print('Loading image features ...')
    img_features_train = prepare_data.get_coco_features(filtered_train)
    print('image features\' shape', img_features_train.shape)


    train_len, num_train_batches = save_batches(img_features_train, questions_train, answers_train, saved_batches_size, 'train')

    del img_features_train
    del questions_train
    del answers_train

    # tokenization and preprocessing for validation question
    questions_val = prepare_data.questions_matrix(filtered_val, word_idx)
    answers_val = prepare_data.answers_matrix(filtered_val, answer_to_onehot_dict)
    img_features_val = prepare_data.get_coco_features(filtered_val)

    val_len, num_val_batches = save_batches(img_features_val, questions_val, answers_val, saved_batches_size, 'val')
    
    del img_features_val
    del questions_val
    del answers_val
    

    print('Creating model ...')

    embedding_matrix = embedding.load()
    if model_name == 'lstm_qi':
        model = models.lstm_qi(num_classes=len(top_answers), vocab_size=len(word_idx)+1, embedding_matrix=embedding_matrix)
    elif model_name == 'lstm_qi_2':
        model = models.lstm_qi_2(num_classes=len(top_answers), vocab_size=len(word_idx)+1, embedding_matrix=embedding_matrix)
    model.summary()

    def print_fn(s):
        with open(f'{model_name}_{dataset}.txt','a') as f:
            print(s, file=f)
    model.summary(print_fn=print_fn)
    plot_model(model, to_file=f'{model_name}_{dataset}.png', show_shapes=True, show_layer_names=True)
    
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


    gen_train = iterate_minibatches(minibatchsize=batch_size, num_saved_batches=num_train_batches, split='train')
    gen_val = iterate_minibatches(minibatchsize=batch_size, num_saved_batches=num_val_batches, split='val')
    model.fit(gen_train,
        epochs=num_epochs,
        steps_per_epoch = math.ceil(train_len/batch_size),
        validation_data=gen_val,
        validation_steps = math.ceil(val_len/batch_size),
        verbose=1)


    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    model.save('weights/' + f'/{model_name}_{dataset}.h5')


if __name__ == '__main__':
    main()
