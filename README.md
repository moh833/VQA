# Visual Question Answering

This is an implementation of visual question answering models from the papers [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf) and [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074)

Trained on [VQA version 1](https://visualqa.org/vqa_v1_download.html) and [COCO-QA](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/) Datasets

* All Models trained with a batch size of 128 and 10 epochs

* For VQA dataset considered only the questions with the 1000 most frequent answers which cover 86% of the data

* Used [GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained word vectors glove.6b with 300D with frozen parameters during training

* Evaluation on VQA dataset done using [VQA Evaluation](https://github.com/GT-Vision-Lab/VQA)

## Datasets

|  | Training Questions | Validation Questions | 
| --- | --- | --- | 
| VQA 1 Dataset | 215264 | 104765 | 
| COCO-QA Dataset | 78736 | 38948 | 

## Models

### BOW
![bow](https://drive.google.com/uc?export=view&id=1VWwCbp5mlCjYdDUHOfpT3gzf1K0MpJ7d)

### VIS_BLSTM
![vis_blstm](https://drive.google.com/uc?export=view&id=1237yA6eGkuNUG6kUk4ddbkj5AL-YmY1l)

### LSTM_QI
![lstm_qi](https://drive.google.com/uc?export=view&id=16zp5wEphpv2tTIRurlAh3WZGxh-dlyaX)

### LSTM_QI_2
![lstm_qi_2](https://drive.google.com/uc?export=view&id=1JmGI1b6LE46FNpAykOwqmfwKxpuqzxkf)

## Validation Accuracy Comparison

| | BOW | VIS BLSTM | LSTM QI | LSTM QI 2 |
| --- | --- | --- | ---| ---|
| VQA 1 | 54.35% | 58.85% | 60.73% | 61.52% |
| COCO-QA  | 49.28% | 54.65% | 55.67% | 55.21% | 

## Training

Using train.py or train_batches.py

train_batches.py splits large data into chunks, and saves them to the hard disk, has an extra argument --chunk_size
```
python train_batches.py --model_name <model_name> --dataset <dataset>
```
* ```-m --model_name``` Name of the model [vis_blstm, lstm_qi, lstm_qi_2]

* ```-d --dataset``` Name of the dataset [VQA_1, COCO-QA]

* ```-cs --chunk_size``` Splits large data into chunks of size chunk_size saved on the hard disk, Default 20480

* ```-ep --epochs``` Default 10

* ```-bz --batch_size``` Default 128

## Prediction

```
python predict_answer.py --image_path <image_path> --question <question>
```
* ```-i --image_path``` Path of the image

* ```-q --question``` The question

* ```-m --model_name``` Default lstm_qi_2

* ```-d --dataset``` Default VQA_1

## Evaluation

```
python Evaluation/evaluate_VQA_1.py --model_name <model_name>
```
* ```-m --model_name``` Name of the model [bow, vis_blstm, lstm_qi, lstm_qi_2]

