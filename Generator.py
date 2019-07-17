import os
GPU = True
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import numpy as np
from tqdm import tqdm

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  parameters
PATH = '/home/zyy/VQA2019/'
MAXLEN = 9

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
def text_standard(text):
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)
model_bert = BertModel.from_pretrained('bert-large-uncased').to(device)
model_bert.eval()

train_imag_list,train_ques_list,train_answ_list = [],[],[]
f = open(PATH + 'ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C4_Abnormality_train.txt', 'r')
lines = f.readlines()
for line in tqdm(lines):
    line = line.strip().split('|')
    word = line[1].split(' ')
    if word[0] not in ['is','are','was','were','do','does','did']:
        train_imag_list.append(line[0])
        train_ques_list.append(text_standard(line[1]))
        train_answ_list.append(text_standard(line[2]))
f.close()
f = open(PATH + 'ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/C4_Abnormality_val.txt', 'r')
lines = f.readlines()
for line in tqdm(lines):
    line = line.strip().split('|')
    word = line[1].split(' ')
    if word[0] not in ['is','are','was','were','do','does','did']:
        train_imag_list.append(line[0])
        train_ques_list.append(text_standard(line[1]))
        train_answ_list.append(text_standard(line[2]))
f.close()
print(len(train_imag_list))

def data_prepare(imag,ques,answ):
    imag_list,ques_list,answ_list,grtr_list = [],[],[],[]
    for i in tqdm(range(len(answ))):
        temp_answ = [1] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answ[i])) + [2]
        for j in range(len(temp_answ)-1):
            grtr_list.append(temp_answ[j+1])
            imag_list.append(imag[i])
            ques_list.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ques[i])))
            sub_answ = temp_answ[:j+1]
            if len(sub_answ)<MAXLEN:
                for k in range(MAXLEN-len(sub_answ)):
                    sub_answ = [0] + sub_answ
            else:
                sub_answ = sub_answ[len(sub_answ)-MAXLEN:]
            answ_list.append(sub_answ)
    return imag_list,ques_list,answ_list,grtr_list
train_imag_list,train_ques_list,train_answ_list,train_grtr_list = data_prepare(train_imag_list,train_ques_list,train_answ_list)
print(len(train_grtr_list))

def get_bert(ques,answ):
    ques_feat,answ_feat = [],[]
    for i in tqdm(range(len(ques))):
        toke_ques = torch.tensor(np.array([(ques[i] + [0] * (MAXLEN - len(ques[i])))[:MAXLEN]]))
        token = torch.cat((toke_ques,torch.tensor([answ[i]])),dim=-1).to(device)
        segm = np.concatenate((np.zeros(MAXLEN, dtype=int),np.ones(MAXLEN, dtype=int)))
        segm = torch.tensor([segm]).to(device)
        with torch.no_grad():
            out, _ = model_bert(token, segm)
        result = np.concatenate((out[-1][0].detach().cpu().numpy(),out[-2][0].detach().cpu().numpy(),out[-3][0].detach().cpu().numpy(),
                                 out[-4][0].detach().cpu().numpy()),axis=-1)
        np.save(PATH + 'data/maga_gene/train_ques_' + str(i), result[:MAXLEN])
        np.save(PATH + 'data/maga_gene/train_answ_' + str(i), result[MAXLEN:])

get_bert(train_ques_list,train_answ_list)
np.save(PATH + 'data/maga_gene/train_grtr', np.array(train_grtr_list))

import keras
from keras.models import Model
from keras.preprocessing import *
from keras.preprocessing.image import *
from keras.applications import *
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)
from keras.applications.nasnet import NASNetLarge, preprocess_input
model_n = NASNetLarge()
model_na = Model(inputs=model_n.input, outputs=model_n.layers[-2].output)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
model_i = InceptionResNetV2()
model_in = Model(inputs=model_i.input, outputs=model_i.layers[-2].output)
from keras.applications.xception import Xception, preprocess_input
model_x = Xception()
model_xc = Model(inputs=model_x.input, outputs=model_x.layers[-2].output)

def get_feat(lists):
    i = 0
    for name in tqdm(lists):
        if os.path.isfile(PATH + 'ImageClef-2019-VQA-Med-Training/Train_images/' + name + '.jpg') == True:
            temp = PATH + 'ImageClef-2019-VQA-Med-Training/Train_images/' + name + '.jpg'
        else:
            temp = PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/' + name + '.jpg'
        temp1 = image.load_img(temp, target_size=(331, 331))
        temp1 = image.img_to_array(temp1)
        temp1 = np.expand_dims(temp1, axis=0)
        for _temp in datagen.flow(temp1, batch_size=1):
                break
        temp1 = nasnet.preprocess_input(_temp)
        temp1 = model_na.predict(temp1)
        temp2 = image.load_img(temp, target_size=(299, 299))
        temp2 = image.img_to_array(temp2)
        temp2 = np.expand_dims(temp2, axis=0)
        for _temp in datagen.flow(temp2, batch_size=1):
                break
        temp2 = inception_resnet_v2.preprocess_input(_temp)
        temp2 = model_in.predict(temp2)
        temp3 = xception.preprocess_input(_temp)
        temp3 = model_xc.predict(temp3)
        temp = np.concatenate((np.squeeze(temp1),np.squeeze(temp2),np.squeeze(temp3)),axis=-1)
        np.save(PATH + 'data/maga_gene/train_imag_' + str(i), temp)
        i += 1

get_feat(train_imag_list)

import os
GPU = True
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import collections
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm

import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.preprocessing import *
from keras.initializers import *
from keras import optimizers
from keras import metrics
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '2019'
random.seed(9102)
np.random.seed(9102)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(9102)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess.run(tf.global_variables_initializer())
K.set_session(sess)

#  parameters
PATH = '/home/zyy/VQA2019/'
MAXLEN = 9

bert2index = {100:0}
count = 1
corpus = np.load(PATH + 'data/maga_gene/train_grtr.npy')
corp_dict = collections.Counter(corpus)
rest_dict = dict(filter(lambda x: x[1]>3, corp_dict.items()))
for i in tqdm(range(len(corpus))):
    if int(corpus[i]) in rest_dict.keys() and int(corpus[i]) not in bert2index.keys():
        bert2index[int(corpus[i])] = count
        count += 1
index2bert ={value:key for key, value in bert2index.items()}
VOCAB = len(bert2index)
print(len(corpus))
print(VOCAB)
def Generator(start,end):
    imag,ques,answ,grtr = [],[],[],[]
    if start < end:
        count = start
        batch = 0
        while count <= end:
            imag.append(np.load(PATH + 'data/maga_gene/train_imag_'+str(count)+'.npy'))
            ques.append(np.load(PATH + 'data/maga_gene/train_ques_'+str(count)+'.npy'))
            answ.append(np.load(PATH + 'data/maga_gene/train_answ_'+str(count)+'.npy'))
            label = corpus[count]
            if label in rest_dict.keys():
                label = bert2index[int(label)]
            else:
                label = 0
            grtr.append(label)
            count += 1
            batch += 1
            if count > end:
                count = start
            if batch == BATCH:
                yield ([np.array(imag),np.array(ques),np.array(answ)],np.expand_dims(np.array(grtr),axis=-1))
                batch = 0
                imag,ques,answ,grtr = [],[],[],[]
    else:
        count,batch = 0,0
        while count < len(corpus):
            imag.append(np.load(PATH + 'data/maga_gene/train_imag_'+str(count)+'.npy'))
            ques.append(np.load(PATH + 'data/maga_gene/train_ques_'+str(count)+'.npy'))
            answ.append(np.load(PATH + 'data/maga_gene/train_answ_'+str(count)+'.npy'))
            label = corpus[count]
            if label in rest_dict.keys():
                label = bert2index[int(label)]
            else:
                label = 0
            grtr.append(label)
            if count == end:
                count = start - 1
            count += 1
            batch += 1
            if count >= len(corpus):
                count = 0
            if batch == BATCH:
                yield ([np.array(imag),np.array(ques),np.array(answ)],np.expand_dims(np.array(grtr),axis=-1))
                batch = 0
                imag,ques,answ,grtr = [],[],[],[]

def get_imgfeat(name,path,pix):
    temp = path + name + '.jpg'
    temp = image.load_img(temp, target_size=(pix, pix))
    temp = image.img_to_array(temp)
    temp = np.expand_dims(temp, axis=0)
    return temp
def text_standard(text):
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)
def get_bert(ques,answ):
    ques_feat,answ_feat = [],[]
    for i in range(len(ques)):
        toke_ques = torch.tensor(np.array([(list(ques[i]) + [0] * (MAXLEN - len(ques[i])))[:MAXLEN]]))
        token = torch.cat((toke_ques,torch.tensor([answ[i]])),dim=-1).to(device)
        segm = np.concatenate((np.zeros(MAXLEN, dtype=int),np.ones(MAXLEN, dtype=int)))
        segm = torch.tensor([segm]).to(device)
        with torch.no_grad():
            out, _ = model_bert(token, segm)
        result = np.concatenate((out[-1][0].detach().cpu().numpy(),out[-2][0].detach().cpu().numpy(),out[-3][0].detach().cpu().numpy(),
                                 out[-4][0].detach().cpu().numpy()),axis=-1)
        ques_feat.append(result[:MAXLEN])
        answ_feat.append(result[MAXLEN:])
    return np.array(ques_feat), np.array(answ_feat)
def get_beam(pair, topk=5):
#     img_path = PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/'
    img_path = PATH + 'VQAMed2019Test/VQAMed2019_Test_Images/'
    feat_na = get_imgfeat(pair[0],img_path,331)
    feat_na = model_na.predict(nasnet.preprocess_input(feat_na))
    feat_in = get_imgfeat(pair[0],img_path,299)
    feat_in = model_in.predict(inception_resnet_v2.preprocess_input(feat_in))
    feat_xc = get_imgfeat(pair[0],img_path,299)
    feat_xc = model_xc.predict(xception.preprocess_input(feat_xc))
    imag_feat = np.reshape(np.array([np.concatenate((feat_na,feat_in,feat_xc),axis=-1)] * topk),(-1,7616))
    ques = text_standard(pair[1])
    ques = np.array([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ques))] * topk)
    answ = np.array([[0,0,0,0,0,0,0,0,1]] * topk)
    score = []
    ques_feat, answ_feat = get_bert(ques, answ)
    prob_1 = model_001.predict([imag_feat, ques_feat, answ_feat])
    prob_2 = model_002.predict([imag_feat, ques_feat, answ_feat])
    prob_3 = model_003.predict([imag_feat, ques_feat, answ_feat])
    prob_4 = model_004.predict([imag_feat, ques_feat, answ_feat])
    prob_5 = model_005.predict([imag_feat, ques_feat, answ_feat])
    prob = prob_1+prob_2+prob_3+prob_4+prob_5
    char_pred = np.argsort(prob)[:,-topk:]
    answ = answ.tolist()
    for i in range(topk):
        answ[i].pop(0)
        answ[i].append(index2bert[char_pred[i][-1-i]])
        score.append(prob[0][char_pred[i][-1-i]])
    answ = np.array(answ)
    temp_score, temp_answ = [], []
    for i in range(MAXLEN-1):
        ques_feat, answ_feat = get_bert(ques, answ)
        prob_1 = model_001.predict([imag_feat, ques_feat, answ_feat])
        prob_2 = model_002.predict([imag_feat, ques_feat, answ_feat])
        prob_3 = model_003.predict([imag_feat, ques_feat, answ_feat])
        prob_4 = model_004.predict([imag_feat, ques_feat, answ_feat])
        prob_5 = model_005.predict([imag_feat, ques_feat, answ_feat])
        prob = prob_1+prob_2+prob_3+prob_4+prob_5
        char_pred = np.argsort(prob)[:,-topk:]
        answ = answ.tolist()
        for j in range(topk):
            for k in range(topk):
                temp = copy.deepcopy(answ[j])
                temp.pop(0)
                temp.append(index2bert[char_pred[j][-1-k]])
                temp_answ.append(temp)
                if temp[-1] not in [temp[-2], 100]:
                    temp_score.append(score[j]+prob[j][char_pred[j][-1-k]])
                else:
                    temp_score.append(score[j]+prob[j][char_pred[j][-1-k]]-0.01)
        for j in range(topk):
            score[j] = temp_score[np.argsort(temp_score)[-1-j]]
            answ[j] = temp_answ[np.argsort(temp_score)[-1-j]]
            answ = np.array(answ)
    result = []
    for i in range(len(answ[0])):
        result.append(tokenizer.convert_ids_to_tokens([answ[0][i]])[0])
    if '[unused1]' in result:
        result = result[:result.index('[unused1]')]
    while len(result)>1 and result[0][0] == '#':
        result.pop(0)
    out = ' '.join(result)
    out = out.replace(' ##','').replace(' [UNK]','')
    if out == '':
        out = 'fracture'
    return out.replace('##','')

# try
DIM = 1024
DROP = 0
BATCH = 512
HE = he_uniform()
# HE = glorot_normal()
L2 = keras.regularizers.l2(1e-6)

in_im = Input(shape=(7616,), name='1')
im = RepeatVector(MAXLEN, name='2')(in_im)
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='3')(im)
in_qu = Input(shape=(MAXLEN,4096), name='4')
qu = Dense(DIM, activation='relu', kernel_initializer=HE, name='5')(in_qu)
in_an = Input(shape=(MAXLEN,4096), name='6')
an = Dense(DIM, activation='relu', kernel_initializer=HE, name='7')(in_an)
me = concatenate([im,qu,an], name='8')
me = LSTM(DIM, return_sequences=False, dropout=DROP, kernel_regularizer=L2, name='9')(me)
me = Dense(VOCAB, activation='softmax', name='10')(me)
EXP = '001'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=9, verbose=1)
call_list = [checkpoint,plateau,stop]
model_001 = Model(inputs=[in_im,in_qu,in_an], outputs=me)
model_001.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_001.fit_generator(Generator(6144,31582), steps_per_epoch=25439//BATCH+1, epochs=9999, callbacks=call_list,
                                  validation_data=Generator(0,6143), validation_steps=6144//BATCH, class_weight='auto')

in_im = Input(shape=(7616,), name='1')
im = RepeatVector(MAXLEN, name='2')(in_im)
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='3')(im)
in_qu = Input(shape=(MAXLEN,4096), name='4')
qu = Dense(DIM, activation='relu', kernel_initializer=HE, name='5')(in_qu)
in_an = Input(shape=(MAXLEN,4096), name='6')
an = Dense(DIM, activation='relu', kernel_initializer=HE, name='7')(in_an)
me = concatenate([im,qu,an], name='8')
me = LSTM(DIM, return_sequences=False, dropout=DROP, kernel_regularizer=L2, name='9')(me)
me = Dense(VOCAB, activation='softmax', name='10')(me)
EXP = '002'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=9, verbose=1)
call_list = [checkpoint,plateau,stop]
model_002 = Model(inputs=[in_im,in_qu,in_an], outputs=me)
model_002.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_002.fit_generator(Generator(12460,6315), steps_per_epoch=25439//BATCH+1, epochs=9999, callbacks=call_list,
                                  validation_data=Generator(6316,12459), validation_steps=6144//BATCH, class_weight='auto')

in_im = Input(shape=(7616,), name='1')
im = RepeatVector(MAXLEN, name='2')(in_im)
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='3')(im)
in_qu = Input(shape=(MAXLEN,4096), name='4')
qu = Dense(DIM, activation='relu', kernel_initializer=HE, name='5')(in_qu)
in_an = Input(shape=(MAXLEN,4096), name='6')
an = Dense(DIM, activation='relu', kernel_initializer=HE, name='7')(in_an)
me = concatenate([im,qu,an], name='8')
me = LSTM(DIM, return_sequences=False, dropout=DROP, kernel_regularizer=L2, name='9')(me)
me = Dense(VOCAB, activation='softmax', name='10')(me)
EXP = '003'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=9, verbose=1)
call_list = [checkpoint,plateau,stop]
model_003 = Model(inputs=[in_im,in_qu,in_an], outputs=me)
model_003.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_003.fit_generator(Generator(18776,12631), steps_per_epoch=25439//BATCH+1, epochs=9999, callbacks=call_list,
                                  validation_data=Generator(12632,18775), validation_steps=6144//BATCH, class_weight='auto')

in_im = Input(shape=(7616,), name='1')
im = RepeatVector(MAXLEN, name='2')(in_im)
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='3')(im)
in_qu = Input(shape=(MAXLEN,4096), name='4')
qu = Dense(DIM, activation='relu', kernel_initializer=HE, name='5')(in_qu)
in_an = Input(shape=(MAXLEN,4096), name='6')
an = Dense(DIM, activation='relu', kernel_initializer=HE, name='7')(in_an)
me = concatenate([im,qu,an], name='8')
me = LSTM(DIM, return_sequences=False, dropout=DROP, kernel_regularizer=L2, name='9')(me)
me = Dense(VOCAB, activation='softmax', name='10')(me)
EXP = '004'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=9, verbose=1)
call_list = [checkpoint,plateau,stop]
model_004 = Model(inputs=[in_im,in_qu,in_an], outputs=me)
model_004.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_004.fit_generator(Generator(25092,18947), steps_per_epoch=25439//BATCH+1, epochs=9999, callbacks=call_list,
                                  validation_data=Generator(18948,25091), validation_steps=6144//BATCH, class_weight='auto')

in_im = Input(shape=(7616,), name='1')
im = RepeatVector(MAXLEN, name='2')(in_im)
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='3')(im)
in_qu = Input(shape=(MAXLEN,4096), name='4')
qu = Dense(DIM, activation='relu', kernel_initializer=HE, name='5')(in_qu)
in_an = Input(shape=(MAXLEN,4096), name='6')
an = Dense(DIM, activation='relu', kernel_initializer=HE, name='7')(in_an)
me = concatenate([im,qu,an], name='8')
me = LSTM(DIM, return_sequences=False, dropout=DROP, kernel_regularizer=L2, name='9')(me)
me = Dense(VOCAB, activation='softmax', name='10')(me)
EXP = '005'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=9, verbose=1)
call_list = [checkpoint,plateau,stop]
model_005 = Model(inputs=[in_im,in_qu,in_an], outputs=me)
model_005.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_005.fit_generator(Generator(31408,25263), steps_per_epoch=25439//BATCH+1, epochs=9999, callbacks=call_list,
                                  validation_data=Generator(25264,31407), validation_steps=6144//BATCH, class_weight='auto')

import copy
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model_bert = BertModel.from_pretrained('bert-large-uncased').to(device)
model_bert.eval()
from keras.preprocessing.image import *
from keras.applications import *
from keras.applications.nasnet import NASNetLarge, preprocess_input
model_n = NASNetLarge()
model_na = Model(inputs=model_n.input, outputs=model_n.layers[-2].output)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
model_i = InceptionResNetV2()
model_in = Model(inputs=model_i.input, outputs=model_i.layers[-2].output)
from keras.applications.xception import Xception, preprocess_input
model_x = Xception()
model_xc = Model(inputs=model_x.input, outputs=model_x.layers[-2].output)
model_001.load_weights(PATH + 'data/weight/001.hdf5', by_name=True)
model_002.load_weights(PATH + 'data/weight/002.hdf5', by_name=True)
model_003.load_weights(PATH + 'data/weight/003.hdf5', by_name=True)
model_004.load_weights(PATH + 'data/weight/004.hdf5', by_name=True)
model_005.load_weights(PATH + 'data/weight/005.hdf5', by_name=True)

f0 = open(PATH + 'data/grtr/try.csv', 'w')
for i in tqdm(range(375,500)):
    f1 = open(PATH + 'VQAMed2019Test/VQAMed2019_Test_Questions.txt', 'r')
    lists = f1.readlines()[i].strip().split('|')
    word = lists[1].split(' ')
    if word[0].lower() not in ['is','are','was','were','do','does','did']:
        f0.write(str(i)+'	'+lists[0]+'	'+get_beam(lists[0:2],topk=10)+'\n')
    else:
        f0.write(str(i)+'	'+lists[0]+'	'+'\n')
    f1.close()
f0.close()