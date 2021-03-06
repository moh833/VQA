import numpy as np
import pandas as pd
import os


def file_to_array(file_path, dtype):
    with open(file_path) as f:
        lines = [line.strip() for line in f]
    return np.asarray(lines, dtype=dtype)


def preprocess(data_path):
    img_ids = file_to_array(data_path + '/train/img_ids.txt', 'int64')
    types = file_to_array(data_path + '/train/types.txt', 'int8')
    questions = file_to_array(data_path + '/train/questions.txt', 'str')
    answers = file_to_array(data_path + '/train/answers.txt', 'str')
    d = {'image_id': img_ids, 'type': types, 'question': questions, 'answer': answers}
    train = pd.DataFrame(data=d)

    img_ids = file_to_array(data_path + '/test/img_ids.txt', 'int64')
    types = file_to_array(data_path + '/test/types.txt', 'int8')
    questions = file_to_array(data_path + '/test/questions.txt', 'str')
    answers = file_to_array(data_path + '/test/answers.txt', 'str')
    d = {'image_id': img_ids, 'type': types, 'question': questions, 'answer': answers}
    test = pd.DataFrame(data=d)

    return train, test


if __name__ == '__main__':
    data_path = 'data/COCO-QA'
    print('processing...')
    train, test = preprocess(data_path)
    processed_path = 'processed_data/COCO-QA'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    train.to_pickle(processed_path + '/train') 
    test.to_pickle(processed_path + '/val') 
    print('saved to', processed_path)