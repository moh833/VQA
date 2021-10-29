import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import embedding
import prepare_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from train_bow import make_matrix
import argparse

os.chdir("..")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=['bow', 'vis_blstm', 'lstm_qi', 'lstm_qi_2'])
    args = parser.parse_args()

    model_name = args.model_name

    dataset = 'COCO-QA'

    model_path = f'weights/{model_name}_{dataset}.h5'
    model = load_model(model_path)

    data_path = f'processed_data/{dataset}'
    val_data = pd.read_pickle(data_path + '/val')

    with open(f'top_answers/{dataset}_top_answers.txt', 'r') as file:
        top_answers = [line.strip() for line in file]
    atoi = {w:i for i,w in enumerate(top_answers)}

    filtered_val = prepare_data.filter_questions(val_data, atoi)

    word_idx = embedding.load_idx('embeddings/' + f'word_idx_{dataset}')

    if model_name == 'bow':
        questions_val = make_matrix(filtered_val, word_idx)
    else:
        questions_val = prepare_data.questions_matrix(filtered_val, word_idx)

    answer_to_onehot_dict = prepare_data.answer_to_onehot(top_answers)
    answers_val = prepare_data.answers_matrix(filtered_val, answer_to_onehot_dict)
    img_features_val = prepare_data.get_coco_features(filtered_val)


    X_val = [img_features_val, questions_val]
    model.summary()

    print("X_val", X_val[0].shape, X_val[1].shape)
    print("y_val", answers_val.shape)


    overall_acc = model.evaluate(X_val,answers_val)[1]
    types = {0: 'object', 1: 'number', 2: 'color', 3: 'location'}

    types_acc = []
    for i in range(4):
        indices = val_data.index[val_data['type'] == i].tolist()
        X_type = []
        X_type.append( X_val[0][indices] )
        X_type.append( X_val[1][indices] )
        y_type = answers_val[indices]
        types_acc.append( model.evaluate(X_type, y_type)[1] )
        print(y_type.shape[0], types[i],  types_acc[i] )


    plt.bar(range(len(types_acc)), types_acc, align='center')
    plt.xticks(range(len(types_acc)), types.values(), rotation='0',fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.savefig(f'Evaluation/{model_name}_{dataset}_acc.png')
    plt.show()

    with open(f'Evaluation/{model_name}_{dataset}_acc.txt', 'w') as file:
        print('Overall: ', overall_acc, file=file)
        for i in range(4):
            print(f'{types[i]}: ', types_acc[i], file=file)

if __name__ == '__main__':
    main()