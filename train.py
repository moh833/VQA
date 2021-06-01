import numpy as np
import pandas as pd
import prepare_data
import models
import sys
import os

def main():

    num_epochs = 10 # 25
    batch_size = 128 # 200
    model_name = 'lstm_qi'
    dataset = 'COCO-QA'

    data_path = f'processed_data/{dataset}'

    train_data = pd.read_pickle(data_path + '/train')
    val_data = pd.read_pickle(data_path + '/test')

    max_answers = 1000
    # get top answers
    top_answers = prepare_data.get_top_answers(max_answers, train_data, val_data)
    atoi = {w:i for i,w in enumerate(top_answers)}
    # itoa = {i+1:w for i,w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    filtered_train = prepare_data.filter_questions(train_data, atoi)
    filtered_val = prepare_data.filter_questions(val_data, atoi)

    print('Loading questions ...')
    # tokenize and preprocess training questions
    questions_train = prepare_data.questions_matrix(filtered_train)
    print('questions\' shape', questions_train.shape)
    # tokenize and preprocess testing questions
    questions_val = prepare_data.questions_matrix(filtered_val)

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
	

    model = models.lstm_qi(num_classes=len(top_answers))
    X_train = [img_features_train, questions_train]
    X_val = [img_features_val, questions_val]
    model.summary()


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
