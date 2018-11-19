import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/27_cate/train/')
DATA_TEST_PATH = os.path.join(DIR_PATH, 'data/27_cate/test/')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'dataJson/data_train.json')
DATA_TEST_JSON = os.path.join(DIR_PATH, 'dataJson/data_test.json')
STOP_WORDS = os.path.join(DIR_PATH, 'stopwords-nlp-vi.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
DICTIONARY_PATH = 'dictionary.txt'
SVM_NOSCALE = os.path.join(DIR_PATH, 'result/svm_')
SVM_SCALE = os.path.join(DIR_PATH, 'result/svm_scale.txt')
KNN_NOSCALE = os.path.join(DIR_PATH, 'result/knn_')
KNN_SCALE = os.path.join(DIR_PATH, 'result/knn_scale.txt')
NB_NOSCALE = os.path.join(DIR_PATH, 'result/nb_')
NB_SCALE = os.path.join(DIR_PATH, 'result/nb_scale.txt')

