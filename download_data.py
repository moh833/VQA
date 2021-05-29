import os
def download_vqa():

    # Download the VQA Images
    # os.system('wget http://images.cocodataset.org/zips/train2014.zip -P zip/')
    # os.system('wget http://images.cocodataset.org/zips/val2014.zip -P zip/')


    # Download the VQA Questions
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip -P zip/')
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip -P zip/')
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip -P zip/')

    # Download the VQA Annotations
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip -P zip/')
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip -P zip/')


    # Unzip
    os.system('unzip zip/v2_Questions_Train_mscoco.zip -d Questions/')
    os.system('unzip zip/v2_Questions_Val_mscoco.zip -d Questions/')
    os.system('unzip zip/v2_Questions_Test_mscoco.zip -d Questions/')
    os.system('unzip zip/v2_Annotations_Train_mscoco.zip -d Annotations/')
    os.system('unzip zip/v2_Annotations_Val_mscoco.zip -d Annotations/')
    os.system('unzip zip/train2014.zip -d Images/')
    # os.system('unzip zip/val2014.zip -d Images/')

    os.system('rm -r zip')

download_vqa()