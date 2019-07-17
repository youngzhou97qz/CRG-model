import os
GPU = False
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import heapq
import random
import numpy as np
import turicreate as tc
import tensorflow as tf
from gensim.summarization import bm25
from tqdm import tqdm

import keras
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '1920'
random.seed(1920)
np.random.seed(1920)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(1920)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess.run(tf.global_variables_initializer())
K.set_session(sess)
tc.config.set_num_gpus(0)

#  parameters
PATH = '/home/zyy/VQA2019/'
DROP = 0.0
DIM = 128
L2 = None

f = open(PATH + 'ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C1_Modality_train.txt', 'r')
lines = f.readlines()
train_name, train_ques, train_answ = [], [], []
for line in lines:
    line = line.strip().split('|')
    train_name.append(line[0])
    train_ques.append(line[1].strip('?').split(' '))
    train_answ.append(line[2])
f.close()

f = open(PATH + 'ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/C1_Modality_val.txt', 'r')
lines = f.readlines()
valid_name, valid_ques, valid_answ = [], [], []
for line in lines:
    line = line.strip().split('|')
    valid_name.append(line[0])
    valid_ques.append(line[1].strip('?').split(' '))
    valid_answ.append(line[2])
f.close()

bm25_model = bm25.BM25(train_ques)
train_imag = tc.image_analysis.load_images(PATH + 'ImageClef-2019-VQA-Med-Training/Train_images/')
train_imag = train_imag.add_row_number()
tc.config.set_num_gpus(0)
similar_model = tc.image_similarity.create(train_imag)
valid_imag = tc.image_analysis.load_images(PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/')
valid_imag = valid_imag.add_row_number()
similar_dic = {}
similarities = similar_model.query(valid_imag[0:len(valid_ques)], k=len(valid_ques))
for i in range(len(valid_ques)):
    similar_dic[valid_imag[i]['path']] = list(similarities['reference_label'])[500*i:500*i+500]
	
#  找最接近图
def imag_retrieval(train_name, test_name, imag, answ, num):
    imag_list = similar_dic[PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/' + test_name[num] + '.jpg']
    ind = train_name.index(imag[imag_list[0]]['path'][63:-4])
    output = answ[ind].strip()
    return output

calc = 0
total = 0
for j in tqdm(range(len(valid_ques))):
    answer = imag_retrieval(train_name, valid_name, train_imag, train_answ, j)
    if valid_answ[j] in ['yes','no']:
        total += 1
        if answer == valid_answ[j]:
            calc += 1
print(calc/total)

#  问题相同+找最接近图
def retrieval(train_name, test_name, ques, imag, answ, num):
    indices, indices_list = [], []
    scores = bm25_model.get_scores(ques[num])
    max_score = heapq.nlargest(1, scores)[0]
    for i in range(len(scores)):
        if scores[i] == max_score:
            indices.append(i)
            indices_list.append(train_name[i] + '.jpg')
    imag_list = similar_dic[PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/' + test_name[num] + '.jpg']
    for i in range(len(imag_list)):
        index = 0
        if imag[imag_list[i]]['path'][63:] in indices_list:
            index = indices_list.index(imag[imag_list[i]]['path'][63:])
            break
    output = answ[indices[index]].strip()
    return output, indices[index], i

calc = 0
total = 0
yes, no = [], []
for j in tqdm(range(len(valid_ques))):
    answer, ind, seq = retrieval(train_name, valid_name, valid_ques, train_imag, train_answ, j)
    if valid_answ[j] in ['yes','no']:
        total += 1
        if answer == valid_answ[j]:
            calc += 1
print(calc/total)