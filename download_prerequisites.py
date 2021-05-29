import os

def download_embeddings(glove_name):
    os.system(f'wget http://nlp.stanford.edu/data/{glove_name}.zip -P embeddings/')
    os.system(f'unzip embeddings/{glove_name}.zip -d embeddings/')
    os.system(f'rm embeddings/{glove_name}.zip')

def download_coco_features():
    os.system('wget https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip')
    os.system('unzip coco.zip')
    os.system('rm coco.zip')
    os.system('mv coco coco_features')

if __name__ == '__main__':
    glove_name = 'glove.6B'
    download_embeddings(glove_name)
    download_coco_features()