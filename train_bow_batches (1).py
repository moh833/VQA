import numpy as np
import pandas as pd
import prepare_data
import models
import os
import math
import embedding
# import errno, os, stat, shutil
import glob
from nltk.tokenize import word_tokenize
import nltk
import pickle
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

def create_vocab(train, threshold = 1, val=np.array([])):
    counts = {}
    questions = train['question'].values.tolist()
    if val.size != 0:
        questions += val['question'].values.tolist()
    for q in questions:
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        words = tokenizer.tokenize(q.lower())
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)

    vocab = [c[1] for c in cw if c[0] > threshold]

    word_idx = {w:i+1 for i, w in enumerate(vocab)}

    word_idx_path = 'embeddings/word_idx'
    with open(word_idx_path,'wb') as f:
        pickle.dump(word_idx,f)
    print(f'vocab = {len(vocab)} words')
    print('top words in questions and their counts:')    
    print('\n'.join(map(str,vocab[:10])))
    return word_idx

def get_vec(data,question):
  tokens = word_tokenize(question)
  vec = np.zeros(len(data)+1) #the zero index is used for error checking "unknown words"
  for word in tokens:
    vec[data.get(word, 0)] += 1 
  return vec

def make_matrix(all_data,qw2i):
  arr = np.empty((len(all_data),len(qw2i)+1),dtype=float)
  for i,q in enumerate(all_data['question']):
    arr[i] = get_vec(qw2i,q.lower())
  return arr

def main():

    num_epochs = 10 
    batch_size = 128 
    saved_batches_size = 20480 # 5120
    model_name = 'bow'
    dataset = 'VQA_1'

    data_path = f'processed_data/{dataset}'

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

    word_idx = create_vocab(filtered_train, 2, filtered_val)

    # save the word_idx
    embedding.save_idx(word_idx, 'embeddings/' + f'word_idx_{dataset}')

    ## load a pre-existing word_idx
    # word_idx = embedding.load_idx('embeddings/' + f'word_idx_{dataset}')

    print('Loading questions ...')
    # tokenization and preprocessing training question
    questions_train = make_matrix(filtered_train, word_idx)
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

    # tokenization and preprocessing testing question
    questions_val = make_matrix(filtered_val, word_idx)
    answers_val = prepare_data.answers_matrix(filtered_val, answer_to_onehot_dict)
    img_features_val = prepare_data.get_coco_features(filtered_val)

    val_len, num_val_batches = save_batches(img_features_val, questions_val, answers_val, saved_batches_size, 'val')
    
    del img_features_val
    del questions_val
    del answers_val
    

    print('Creating model ...')

    model = models.bow_model(num_classes=len(top_answers), vocab_size=len(word_idx)+1)
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
