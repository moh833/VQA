# coding: utf-8
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare_data
import models
import embedding
from keras.models import load_model, Model
import json
from nltk.tokenize import word_tokenize
import argparse

os.chdir("..")

def preprocess(split):
    assert(split in ['train', 'val'])

    # only the data with 1000 most frequent answers
    anno = json.load(open(f'Evaluation/Annotations/reduced_mscoco_{split}2014_annotations.json', 'r'))
    ques = json.load(open(f'Evaluation/Questions/OpenEnded_reduced_mscoco_{split}2014_questions.json', 'r'))

    data_len = len(anno['annotations'])
    answers = []
    image_ids = []
    questions = []
    question_ids = []
    for i in range(data_len):
        answers.append( anno['annotations'][i]['multiple_choice_answer'] )
        image_ids.append( anno['annotations'][i]['image_id'] )
        questions.append( ques['questions'][i]['question'] )
        question_ids.append( ques['questions'][i]['question_id'] )
    answers = np.asarray(answers, dtype='str')
    image_ids = np.asarray(image_ids, dtype='int64')
    questions = np.asarray(questions, dtype='str')
    question_ids = np.asarray(question_ids, dtype='int64')

    d = {'image_id': image_ids, 'question_id': question_ids, 'question': questions, 'answer': answers}
    data = pd.DataFrame(data=d)

    return data

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

def dump_results_file(model_name, data_path):

    dataset = 'VQA_1'
    split = 'val'

    val_data = preprocess(split)

    with open(f'top_answers/{dataset}_top_answers.txt', 'r') as file:
        top_answers = [line.strip() for line in file]

    atoi = {w:i for i,w in enumerate(top_answers)}
    
    word_idx = embedding.load_idx('embeddings/' + f'/word_idx_{dataset}')

    print('Loading questions ...')
    if model_name == 'bow':
        questions_val = make_matrix(val_data, word_idx)
    else:
        questions_val = prepare_data.questions_matrix(val_data, word_idx)
        

    print('Loading answers ...')
    answer_to_onehot_dict = prepare_data.answer_to_onehot(top_answers)
    answers_val = prepare_data.answers_matrix(val_data, answer_to_onehot_dict)


    print('Loading image features ...')
    img_features_val = prepare_data.get_coco_features(val_data)

    X_val = [img_features_val, questions_val]

    print("X_val", X_val[0].shape, X_val[1].shape)
    print("y_val", answers_val.shape)

    model_path = f'weights/{model_name}_{dataset}.h5'
    model = load_model(model_path)

    print( "validation loss, accuracy", model.evaluate(X_val, answers_val) )

    probabilities = model.predict(X_val)
	
    answers_ids = np.argmax(probabilities, axis=-1)

    predicted_answers = []
    for ans_id in answers_ids:
        predicted_answers.append( top_answers[ans_id] )


    q_ids = val_data['question_id'].values.tolist()
    data_len = len(q_ids)
    json_results = []
    for i in range(data_len):
        json_results.append({"answer": predicted_answers[i], "question_id": q_ids[i]})

    if not os.path.exists(data_path + '/Results/'):
        os.makedirs(data_path + '/Results/')
        
    with open(data_path + f'/Results/OpenEnded_reduced_mscoco_{split}2014_{model_name}_results.json', 'w') as file:
        json.dump(json_results, file)
	

from PythonHelperTools.vqaTools.vqa import VQA
from PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def main():
    dataDir = os.getcwd() + '/Evaluation'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=['bow', 'vis_blstm', 'lstm_qi', 'lstm_qi_2'])
    args = parser.parse_args()

    dump_results_file(args.model_name, dataDir)

    data_name = 'reduced_mscoco_val2014'
    annFile     ='%s/Annotations/%s_annotations.json'%(dataDir, data_name)
    quesFile    ='%s/Questions/OpenEnded_%s_questions.json'%(dataDir, data_name)

    resultType  = args.model_name

    fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

    [resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/OpenEnded_%s_%s_%s.json'%(dataDir, data_name, \
    resultType, fileType) for fileType in fileTypes]  

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    # plot accuracy for various question types
    figure(figsize=(20, 8))

    plt.tight_layout()
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='90',fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.savefig(dataDir + f'/Results/{resultType}_graph.png')
    plt.show()


    # save evaluation results to ./Results folder
    json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))


if __name__ == '__main__':
    main()