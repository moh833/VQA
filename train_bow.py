import numpy as np
import pandas as pd
import prepare_data
import models
import os
import embedding
from nltk.tokenize import word_tokenize
import nltk
from keras.utils.vis_utils import plot_model
import argparse

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['COCO-QA', 'VQA_1'], help="The dataset to train on")
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bz', '--batch_size', type=int, default=128)
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    dataset = args.dataset
    model_name = 'bow'

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

    word_idx = create_vocab(filtered_train, 2)

    # save the word_idx
    embedding.save_idx(word_idx, 'embeddings/' + f'word_idx_{dataset}')

    ## load a pre-existing word_idx
    # word_idx = embedding.load_idx('embeddings/' + f'word_idx_{dataset}')

    print('Loading questions ...')
    # tokenize and preprocess training questions
    questions_train = make_matrix(filtered_train, word_idx)
    print('questions\' shape', questions_train.shape)
    # tokenize and preprocess testing questions
    questions_val = make_matrix(filtered_val, word_idx)

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
	
    model = models.bow_model(num_classes=len(top_answers), vocab_size=len(word_idx)+1)

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
