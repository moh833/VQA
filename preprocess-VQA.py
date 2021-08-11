import json
import numpy as np
import pandas as pd
import os

def preprocess(data_path, split):
    assert(split in ['train', 'val'])
    
    anno = json.load(open(data_path + f'/Annotations/mscoco_{split}2014_annotations.json', 'r'))
    ques = json.load(open(data_path + f'/Questions/OpenEnded_mscoco_{split}2014_questions.json', 'r'))
    
    data_len = len(anno['annotations'])
    answers = []
    image_ids = []
    questions = []
    for i in range(data_len):
        answers.append( anno['annotations'][i]['multiple_choice_answer'] )
        image_ids.append( anno['annotations'][i]['image_id'] )
        questions.append( ques['questions'][i]['question'] )
    answers = np.asarray(answers, dtype='str')
    image_ids = np.asarray(image_ids, dtype='int64')
    questions = np.asarray(questions, dtype='str')

    d = {'image_id': image_ids, 'question': questions, 'answer': answers}
    data = pd.DataFrame(data=d)
    
    return data


if __name__ == '__main__':
    data_path = 'data/VQA_1'
    print('processing...')
    train = preprocess(data_path, 'train')
    val = preprocess(data_path, 'val')

    processed_path = 'processed_data/VQA_1'
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    train.to_pickle(processed_path + '/train') 
    val.to_pickle(processed_path + '/val') 
    print('saved to', processed_path)