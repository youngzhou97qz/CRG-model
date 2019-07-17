import re
from tqdm import tqdm
import numpy as np
import keras
from keras.preprocessing import *
from keras.preprocessing.image import *
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model_bert = BertModel.from_pretrained('bert-large-uncased').to(device)
model_bert.eval()

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

PIX = 96
MAXLEN = 9
PATH = '/home/zyy/VQA2019/'
def text_standard(text):
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)
	
# tuxiang
f = open(PATH + 'ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C1_Modality_train.txt','r',encoding='utf-8')
tuxiang_train_imag,tuxiang_train_ques,tuxiang_train_answ = [],[],[]
lines = f.readlines()
for line in lines:
    line = line.strip().split('|')
    if line[2] in ['xr - plain film','ct noncontrast','ct with iv contrast','cta - ct angiography','ct w/contrast (iv)',
                   'ct with gi and iv contrast','ct - gi & iv contrast','ct with gi contrast','ct - myelogram','pet-ct fusion',
                   'mr - t2 weighted','mr - t1w - noncontrast','mr - t1w w/gadolinium','mr - dwi diffusion weighted',
                   'mra - mr angiography/venography','mr - other pulse seq.','mr - pdw proton density','mr - adc map (app diff coeff)',
                   'mr - t1w w/gd (fat suppressed)','mr t2* gradient,gre,mpgr,swan,swi','mr - stir','mr - flair w/gd','mr - fiesta',
                   'us - ultrasound','us-d - doppler ultrasound','mammograph','bas - barium swallow','ugi - upper gi','be - barium enema',
                   'sbft - small bowel','an - angiogram','venogram','nm - nuclear medicine','pet - positron emission','mr - flair']:
        tuxiang_train_imag.append(PATH + 'ImageClef-2019-VQA-Med-Training/Train_images/' + line[0] + '.jpg')
        tuxiang_train_ques.append(text_standard(line[1]))
        if line[2] ==  'xr - plain film':
            tuxiang_train_answ.append(0)
        elif line[2] == 'ct noncontrast':
            tuxiang_train_answ.append(1)
        elif line[2] == 'ct with iv contrast':
            tuxiang_train_answ.append(2)
        elif line[2] == 'cta - ct angiography':
            tuxiang_train_answ.append(3)
        elif line[2] == 'ct w/contrast (iv)':
            tuxiang_train_answ.append(4)
        elif line[2] == 'ct with gi and iv contrast':
            tuxiang_train_answ.append(5)
        elif line[2] == 'ct - gi & iv contrast':
            tuxiang_train_answ.append(6)
        elif line[2] == 'ct with gi contrast':
            tuxiang_train_answ.append(7)
        elif line[2] == 'ct - myelogram':
            tuxiang_train_answ.append(8)
        elif line[2] == 'pet-ct fusion':
            tuxiang_train_answ.append(9)
        elif line[2] == 'mr - t2 weighted':
            tuxiang_train_answ.append(10)
        elif line[2] == 'mr - t1w - noncontrast':
            tuxiang_train_answ.append(11)
        elif line[2] == 'mr - t1w w/gadolinium':
            tuxiang_train_answ.append(12)
        elif line[2] == 'mr - dwi diffusion weighted':
            tuxiang_train_answ.append(13)
        elif line[2] == 'mra - mr angiography/venography':
            tuxiang_train_answ.append(14)
        elif line[2] == 'mr - other pulse seq.':
            tuxiang_train_answ.append(15)
        elif line[2] == 'mr - pdw proton density':
            tuxiang_train_answ.append(16)
        elif line[2] == 'mr - adc map (app diff coeff)':
            tuxiang_train_answ.append(17)
        elif line[2] == 'mr - t1w w/gd (fat suppressed)':
            tuxiang_train_answ.append(18)
        elif line[2] == 'mr t2* gradient,gre,mpgr,swan,swi':
            tuxiang_train_answ.append(19)
        elif line[2] == 'mr - stir':
            tuxiang_train_answ.append(20)
        elif line[2] == 'mr - flair w/gd':
            tuxiang_train_answ.append(21)
        elif line[2] == 'mr - fiesta':
            tuxiang_train_answ.append(22)
        elif line[2] == 'us - ultrasound':
            tuxiang_train_answ.append(23)
        elif line[2] == 'us-d - doppler ultrasound':
            tuxiang_train_answ.append(24)
        elif line[2] == 'mammograph':
            tuxiang_train_answ.append(25)
        elif line[2] == 'bas - barium swallow':
            tuxiang_train_answ.append(26)
        elif line[2] == 'ugi - upper gi':
            tuxiang_train_answ.append(27)
        elif line[2] == 'be - barium enema':
            tuxiang_train_answ.append(28)
        elif line[2] == 'sbft - small bowel':
            tuxiang_train_answ.append(29)
        elif line[2] == 'an - angiogram':
            tuxiang_train_answ.append(30)
        elif line[2] == 'venogram':
            tuxiang_train_answ.append(31)
        elif line[2] == 'nm - nuclear medicine':
            tuxiang_train_answ.append(32)
        elif line[2] == 'pet - positron emission':
            tuxiang_train_answ.append(33)
        elif line[2] == 'mr - flair':
            tuxiang_train_answ.append(34)
f.close()
f = open(PATH + 'ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/C1_Modality_val.txt','r',encoding='utf-8')
tuxiang_valid_imag,tuxiang_valid_ques,tuxiang_valid_answ = [],[],[]
lines = f.readlines()
for line in lines:
    line = line.strip().split('|')
    if line[2] in ['xr - plain film','ct noncontrast','ct with iv contrast','cta - ct angiography','ct w/contrast (iv)',
                   'ct with gi and iv contrast','ct - gi & iv contrast','ct with gi contrast','ct - myelogram','pet-ct fusion',
                   'mr - t2 weighted','mr - t1w - noncontrast','mr - t1w w/gadolinium','mr - dwi diffusion weighted',
                   'mra - mr angiography/venography','mr - other pulse seq.','mr - pdw proton density','mr - adc map (app diff coeff)',
                   'mr - t1w w/gd (fat suppressed)','mr t2* gradient,gre,mpgr,swan,swi','mr - stir','mr - flair w/gd','mr - fiesta',
                   'us - ultrasound','us-d - doppler ultrasound','mammograph','bas - barium swallow','ugi - upper gi','be - barium enema',
                   'sbft - small bowel','an - angiogram','venogram','nm - nuclear medicine','pet - positron emission','mr - flair']:
        tuxiang_valid_imag.append(PATH + 'ImageClef-2019-VQA-Med-Validation/Val_images/' + line[0] + '.jpg')
        tuxiang_valid_ques.append(text_standard(line[1]))
        if line[2] ==  'xr - plain film':
            tuxiang_valid_answ.append(0)
        elif line[2] == 'ct noncontrast':
            tuxiang_valid_answ.append(1)
        elif line[2] == 'ct with iv contrast':
            tuxiang_valid_answ.append(2)
        elif line[2] == 'cta - ct angiography':
            tuxiang_valid_answ.append(3)
        elif line[2] == 'ct w/contrast (iv)':
            tuxiang_valid_answ.append(4)
        elif line[2] == 'ct with gi and iv contrast':
            tuxiang_valid_answ.append(5)
        elif line[2] == 'ct - gi & iv contrast':
            tuxiang_valid_answ.append(6)
        elif line[2] == 'ct with gi contrast':
            tuxiang_valid_answ.append(7)
        elif line[2] == 'ct - myelogram':
            tuxiang_valid_answ.append(8)
        elif line[2] == 'pet-ct fusion':
            tuxiang_valid_answ.append(9)
        elif line[2] == 'mr - t2 weighted':
            tuxiang_valid_answ.append(10)
        elif line[2] == 'mr - t1w - noncontrast':
            tuxiang_valid_answ.append(11)
        elif line[2] == 'mr - t1w w/gadolinium':
            tuxiang_valid_answ.append(12)
        elif line[2] == 'mr - dwi diffusion weighted':
            tuxiang_valid_answ.append(13)
        elif line[2] == 'mra - mr angiography/venography':
            tuxiang_valid_answ.append(14)
        elif line[2] == 'mr - other pulse seq.':
            tuxiang_valid_answ.append(15)
        elif line[2] == 'mr - pdw proton density':
            tuxiang_valid_answ.append(16)
        elif line[2] == 'mr - adc map (app diff coeff)':
            tuxiang_valid_answ.append(17)
        elif line[2] == 'mr - t1w w/gd (fat suppressed)':
            tuxiang_valid_answ.append(18)
        elif line[2] == 'mr t2* gradient,gre,mpgr,swan,swi':
            tuxiang_valid_answ.append(19)
        elif line[2] == 'mr - stir':
            tuxiang_valid_answ.append(20)
        elif line[2] == 'mr - flair w/gd':
            tuxiang_valid_answ.append(21)
        elif line[2] == 'mr - fiesta':
            tuxiang_valid_answ.append(22)
        elif line[2] == 'us - ultrasound':
            tuxiang_valid_answ.append(23)
        elif line[2] == 'us-d - doppler ultrasound':
            tuxiang_valid_answ.append(24)
        elif line[2] == 'mammograph':
            tuxiang_valid_answ.append(25)
        elif line[2] == 'bas - barium swallow':
            tuxiang_valid_answ.append(26)
        elif line[2] == 'ugi - upper gi':
            tuxiang_valid_answ.append(27)
        elif line[2] == 'be - barium enema':
            tuxiang_valid_answ.append(28)
        elif line[2] == 'sbft - small bowel':
            tuxiang_valid_answ.append(29)
        elif line[2] == 'an - angiogram':
            tuxiang_valid_answ.append(30)
        elif line[2] == 'venogram':
            tuxiang_valid_answ.append(31)
        elif line[2] == 'nm - nuclear medicine':
            tuxiang_valid_answ.append(32)
        elif line[2] == 'pet - positron emission':
            tuxiang_valid_answ.append(33)
        elif line[2] == 'mr - flair':
            tuxiang_valid_answ.append(34)
f.close()
f = open(PATH + 'VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt','r',encoding='utf-8')
tuxiang_test_imag,tuxiang_test_ques,tuxiang_test_answ = [],[],[]
lines = f.readlines()[0:125]
for line in lines:
    line = line.strip().split('|')
    if line[3] in ['xr - plain film','ct noncontrast','ct with iv contrast','cta - ct angiography','ct w/contrast (iv)',
                   'ct with gi and iv contrast','ct - gi & iv contrast','ct with gi contrast','ct - myelogram','pet-ct fusion',
                   'mr - t2 weighted','mr - t1w - noncontrast','mr - t1w w/gadolinium','mr - dwi diffusion weighted',
                   'mra - mr angiography/venography','mr - other pulse seq.','mr - pdw proton density','mr - adc map (app diff coeff)',
                   'mr - t1w w/gd (fat suppressed)','mr t2* gradient,gre,mpgr,swan,swi','mr - stir','mr - flair w/gd','mr - fiesta',
                   'us - ultrasound','us-d - doppler ultrasound','mammograph','bas - barium swallow','ugi - upper gi','be - barium enema',
                   'sbft - small bowel','an - angiogram','venogram','nm - nuclear medicine','pet - positron emission','mr - flair']:
        tuxiang_test_imag.append(PATH + 'VQAMed2019Test/VQAMed2019_Test_Images/' + line[0] + '.jpg')
        tuxiang_test_ques.append(text_standard(line[2]))
        if line[3] ==  'xr - plain film':
            tuxiang_test_answ.append(0)
        elif line[3] == 'ct noncontrast':
            tuxiang_test_answ.append(1)
        elif line[3] == 'ct with iv contrast':
            tuxiang_test_answ.append(2)
        elif line[3] == 'cta - ct angiography':
            tuxiang_test_answ.append(3)
        elif line[3] == 'ct w/contrast (iv)':
            tuxiang_test_answ.append(4)
        elif line[3] == 'ct with gi and iv contrast':
            tuxiang_test_answ.append(5)
        elif line[3] == 'ct - gi & iv contrast':
            tuxiang_test_answ.append(6)
        elif line[3] == 'ct with gi contrast':
            tuxiang_test_answ.append(7)
        elif line[3] == 'ct - myelogram':
            tuxiang_test_answ.append(8)
        elif line[3] == 'pet-ct fusion':
            tuxiang_test_answ.append(9)
        elif line[3] == 'mr - t2 weighted':
            tuxiang_test_answ.append(10)
        elif line[3] == 'mr - t1w - noncontrast':
            tuxiang_test_answ.append(11)
        elif line[3] == 'mr - t1w w/gadolinium':
            tuxiang_test_answ.append(12)
        elif line[3] == 'mr - dwi diffusion weighted':
            tuxiang_test_answ.append(13)
        elif line[3] == 'mra - mr angiography/venography':
            tuxiang_test_answ.append(14)
        elif line[3] == 'mr - other pulse seq.':
            tuxiang_test_answ.append(15)
        elif line[3] == 'mr - pdw proton density':
            tuxiang_test_answ.append(16)
        elif line[3] == 'mr - adc map (app diff coeff)':
            tuxiang_test_answ.append(17)
        elif line[3] == 'mr - t1w w/gd (fat suppressed)':
            tuxiang_test_answ.append(18)
        elif line[3] == 'mr t2* gradient,gre,mpgr,swan,swi':
            tuxiang_test_answ.append(19)
        elif line[3] == 'mr - stir':
            tuxiang_test_answ.append(20)
        elif line[3] == 'mr - flair w/gd':
            tuxiang_test_answ.append(21)
        elif line[3] == 'mr - fiesta':
            tuxiang_test_answ.append(22)
        elif line[3] == 'us - ultrasound':
            tuxiang_test_answ.append(23)
        elif line[3] == 'us-d - doppler ultrasound':
            tuxiang_test_answ.append(24)
        elif line[3] == 'mammograph':
            tuxiang_test_answ.append(25)
        elif line[3] == 'bas - barium swallow':
            tuxiang_test_answ.append(26)
        elif line[3] == 'ugi - upper gi':
            tuxiang_test_answ.append(27)
        elif line[3] == 'be - barium enema':
            tuxiang_test_answ.append(28)
        elif line[3] == 'sbft - small bowel':
            tuxiang_test_answ.append(29)
        elif line[3] == 'an - angiogram':
            tuxiang_test_answ.append(30)
        elif line[3] == 'venogram':
            tuxiang_test_answ.append(31)
        elif line[3] == 'nm - nuclear medicine':
            tuxiang_test_answ.append(32)
        elif line[3] == 'pet - positron emission':
            tuxiang_test_answ.append(33)
        elif line[3] == 'mr - flair':
            tuxiang_test_answ.append(34)
f.close()

def get_stack(lists):
    stack = []
    for name in tqdm(lists):
        temp1 = image.load_img(name, target_size=(331, 331))
        temp1 = image.img_to_array(temp1)
        temp1 = np.expand_dims(temp1, axis=0)
        for _temp in datagen.flow(temp1, batch_size=1):
                break
        temp1 = nasnet.preprocess_input(_temp)
        temp1 = model_na.predict(temp1)
        temp2 = image.load_img(name, target_size=(299, 299))
        temp2 = image.img_to_array(temp2)
        temp2 = np.expand_dims(temp2, axis=0)
        for _temp in datagen.flow(temp2, batch_size=1):
                break
        temp2 = inception_resnet_v2.preprocess_input(_temp)
        temp2 = model_in.predict(temp2)
        temp3 = xception.preprocess_input(_temp)
        temp3 = model_xc.predict(temp3)
        temp = np.concatenate((np.squeeze(temp1),np.squeeze(temp2),np.squeeze(temp3)),axis=-1)
        stack.append(np.squeeze(temp))
    return np.array(stack)
tuxiang_train_stack = get_stack(tuxiang_train_imag)
tuxiang_valid_stack = get_stack(tuxiang_valid_imag)
tuxiang_test_stack = get_stack(tuxiang_test_imag)
np.save(PATH + 'data/maga_clas/tuxiang_train_stack', tuxiang_train_stack)
np.save(PATH + 'data/maga_clas/tuxiang_valid_stack', tuxiang_valid_stack)
np.save(PATH + 'data/maga_clas/tuxiang_test_stack', tuxiang_test_stack)

def get_feat(lists):
    feat = []
    for name in tqdm(lists):
        temp = image.load_img(name, target_size=(PIX, PIX))
        temp = image.img_to_array(temp)
        temp = np.expand_dims(temp, axis=0)
        feat.append(np.squeeze(temp))
    return np.array(feat)
tuxiang_train_feat = get_feat(tuxiang_train_imag)
tuxiang_valid_feat = get_feat(tuxiang_valid_imag)
tuxiang_test_feat = get_feat(tuxiang_test_imag)
np.save(PATH + 'data/maga_clas/tuxiang_train_feat', tuxiang_train_feat)
np.save(PATH + 'data/maga_clas/tuxiang_valid_feat', tuxiang_valid_feat)
np.save(PATH + 'data/maga_clas/tuxiang_test_feat', tuxiang_test_feat)

def get_bert(ques):
    ques_id, ques_feat = [],[]
    for i in range(len(ques)):
        ques_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ques[i])))
    ques_id = np.array(ques_id)
    for i in tqdm(range(len(ques_id))):
        token = torch.tensor(np.array([(ques_id[i] + [0] * (MAXLEN - len(ques_id[i])))[:MAXLEN]])).to(device)
        segm = torch.tensor([np.zeros(MAXLEN, dtype=int)]).to(device)
        with torch.no_grad():
            out, _ = model_bert(token, segm)
        result = np.concatenate((out[-1][0].detach().cpu().numpy(),out[-2][0].detach().cpu().numpy(),out[-3][0].detach().cpu().numpy(),
                                 out[-4][0].detach().cpu().numpy()),axis=-1)
        ques_feat.append(result)
    return np.array(ques_feat)
tuxiang_train_bert = get_bert(tuxiang_train_ques)
tuxiang_valid_bert = get_bert(tuxiang_valid_ques)
tuxiang_test_bert = get_bert(tuxiang_test_ques)
np.save(PATH + 'data/maga_clas/tuxiang_train_bert', tuxiang_train_bert)
np.save(PATH + 'data/maga_clas/tuxiang_valid_bert', tuxiang_valid_bert)
np.save(PATH + 'data/maga_clas/tuxiang_test_bert', tuxiang_test_bert)

tuxiang_train_glove = np.zeros((len(tuxiang_train_ques),MAXLEN,300))
tuxiang_valid_glove = np.zeros((len(tuxiang_valid_ques),MAXLEN,300))
tuxiang_test_glove = np.zeros((len(tuxiang_test_ques),MAXLEN,300))
f = open('/home/zyy/VQA/data/W2V/glove.42B.300d.txt','r',encoding='utf-8')
lines = f.readlines()
for line in tqdm(lines):
    label = ''.join(line.split(' ')[:-300])
    glove = line.split(' ')[-300:]
    for i in range(len(tuxiang_train_ques)):
        words = tuxiang_train_ques[i].split(' ')[:MAXLEN]
        for j in range(len(words)):
            if words[j] == label:
                tuxiang_train_glove[i][j] = glove
    for i in range(len(tuxiang_valid_ques)):
        words = tuxiang_valid_ques[i].split(' ')[:MAXLEN]
        for j in range(len(words)):
            if words[j] == label:
                tuxiang_valid_glove[i][j] = glove
    for i in range(len(tuxiang_test_ques)):
        words = tuxiang_test_ques[i].split(' ')[:MAXLEN]
        for j in range(len(words)):
            if words[j] == label:
                tuxiang_test_glove[i][j] = glove
f.close()
np.save(PATH + 'data/maga_clas/tuxiang_train_glove', tuxiang_train_glove)
np.save(PATH + 'data/maga_clas/tuxiang_valid_glove', tuxiang_valid_glove)
np.save(PATH + 'data/maga_clas/tuxiang_test_glove', tuxiang_test_glove)

np.save(PATH + 'data/maga_clas/shifei_train_answ', np.array(shifei_train_answ))
np.save(PATH + 'data/maga_clas/shifei_valid_answ', np.array(shifei_valid_answ))
np.save(PATH + 'data/maga_clas/shifei_test_answ', np.array(shifei_test_answ))
np.save(PATH + 'data/maga_clas/tuxiang_train_answ', np.array(tuxiang_train_answ))
np.save(PATH + 'data/maga_clas/tuxiang_valid_answ', np.array(tuxiang_valid_answ))
np.save(PATH + 'data/maga_clas/tuxiang_test_answ', np.array(tuxiang_test_answ))
np.save(PATH + 'data/maga_clas/qiguan_train_answ', np.array(qiguan_train_answ))
np.save(PATH + 'data/maga_clas/qiguan_valid_answ', np.array(qiguan_valid_answ))
np.save(PATH + 'data/maga_clas/qiguan_test_answ', np.array(qiguan_test_answ))
np.save(PATH + 'data/maga_clas/pingmian_train_answ', np.array(pingmian_train_answ))
np.save(PATH + 'data/maga_clas/pingmian_valid_answ', np.array(pingmian_valid_answ))
np.save(PATH + 'data/maga_clas/pingmian_test_answ', np.array(pingmian_test_answ))

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
random.seed(2019)
np.random.seed(2019)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(2019)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess.run(tf.global_variables_initializer())
K.set_session(sess)

# def rec_unit(x, size, stage):
#     x = Conv2D(size, 3, activation='relu', name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x)
#     x = BatchNormalization(name='norm'+stage+'_1')(x)
#     x = Dropout(DROP, name='drop'+stage+'_1')(x)
#     x1 = Conv2D(size, 3, activation='relu', name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x)
#     x1 = BatchNormalization(name='norm'+stage+'_2')(x1)
#     x1 = Dropout(DROP, name='drop'+stage+'_2')(x1)
#     x1 = add([x,x1], name='add'+stage+'_1')
#     x1 = Conv2D(size, 3, activation='relu', name='conv'+stage+'_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x1)
#     x1 = BatchNormalization(name='norm'+stage+'_3')(x1)
#     x1 = Dropout(DROP, name='drop'+stage+'_3')(x1)
#     return x1

# def att_unit(x, size, stage):
#     a, b = x
#     a = Conv2D(size, 1, activation='relu', name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(a)
#     a = BatchNormalization(name='norm'+stage+'_1')(a)
#     a = Dropout(DROP, name='drop'+stage+'_1')(a)
#     b1 = Conv2D(size, 1, activation='relu', name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(b)
#     b1 = BatchNormalization(name='norm'+stage+'_2')(b1)
#     b1 = Dropout(DROP, name='drop'+stage+'_2')(b1)
#     a = add([a,b1], name='add'+stage+'_1')
#     a = Conv2D(size, 1, activation='relu', name='conv'+stage+'_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(a)
#     a = BatchNormalization(name='norm'+stage+'_3')(a)
#     a = Dropout(DROP, name='drop'+stage+'_3')(a)
#     a = Activation('sigmoid', name='sigm'+stage+'_1')(a)
#     a = multiply([a,b], name='mul'+stage+'_1')
#     return a

# def U_Net(x):
#     size = [DIM//32,DIM//16,DIM//8,DIM//4]
#     x1 = rec_unit(x, stage='x1', size=size[0])
#     x2 = MaxPooling2D((2, 2), strides=2, name='pool1')(x1)
#     x2 = rec_unit(x2, stage='x2', size=size[1])
#     x3 = MaxPooling2D((2, 2), strides=2, name='pool2')(x2)
#     x3 = rec_unit(x3, stage='x3', size=size[2])
#     x4 = MaxPooling2D((2, 2), strides=2, name='pool3')(x3)
#     x4 = rec_unit(x4, stage='x4', size=size[3])
#     d3 = Conv2DTranspose(size[2], 2, strides=2, name='up3', padding='same')(x4)
#     d3 = BatchNormalization(name='n3')(d3)
#     x3 = att_unit([d3, x3], stage='a3', size=size[2])
#     d3 = concatenate([x3, d3], name='cat3', axis=3)
#     d3 = rec_unit(d3, stage='d3', size=size[2])
#     d2 = Conv2DTranspose(size[1], 2, strides=2, name='up2', padding='same')(d3)
#     d2 = BatchNormalization(name='n2')(d2)
#     x2 = att_unit([d2, x2], stage='a2', size=size[1])
#     d2 = concatenate([x2, d2], name='cat2', axis=3)
#     d2 = rec_unit(d2, stage='d2', size=size[1])
#     d1 = Conv2DTranspose(size[0], 2, strides=2, name='up1', padding='same')(d2)
#     d1 = BatchNormalization(name='n1')(d1)
#     x1 = att_unit([d1, x1], stage='a1', size=size[0])
#     d1 = concatenate([x1, d1], name='cat1', axis=3)
#     d1 = rec_unit(d1, stage='d1', size=size[0])
#     u2 = MaxPooling2D((2, 2), strides=2, name='pool5')(d1)
#     u2 = rec_unit(u2, stage='u2', size=size[1])
#     d2 = att_unit([u2, d2], stage='c2', size=size[1])
#     u2 = concatenate([d2, u2], name='con2', axis=3)
#     u3 = MaxPooling2D((2, 2), strides=2, name='pool6')(u2)
#     u3 = rec_unit(u3, stage='u3', size=size[2])
#     d3 = att_unit([u3, d3], stage='c3', size=size[2])
#     u3 = concatenate([d3, u3], name='con3', axis=3)
#     u4 = MaxPooling2D((2, 2), strides=2, name='pool7')(u3)
#     u4 = rec_unit(u4, stage='u4', size=size[3])
#     x4 = att_unit([u4, x4], stage='c4', size=size[3])
#     u4 = concatenate([x4, u4], name='con4', axis=3)
#     u4 = GlobalAveragePooling2D(name='out')(u4)
#     return u4

# # unet
# DIM = 512
# DROP = 0.0
# BATCH = 64
# HE = he_uniform()
# L2 = keras.regularizers.l2(1e-6)
# PIX = 96

# tuxiang
tuxiang_train_stack = np.load(PATH + 'data/maga_clas/tuxiang_train_stack.npy')
tuxiang_train_glove = np.load(PATH + 'data/maga_clas/tuxiang_train_glove.npy')
tuxiang_train_answ = np.load(PATH + 'data/maga_clas/tuxiang_train_answ.npy')
tuxiang_valid_stack = np.load(PATH + 'data/maga_clas/tuxiang_valid_stack.npy')
tuxiang_valid_glove = np.load(PATH + 'data/maga_clas/tuxiang_valid_glove.npy')
tuxiang_valid_answ = np.load(PATH + 'data/maga_clas/tuxiang_valid_answ.npy')
tuxiang_test_stack = np.load(PATH + 'data/maga_clas/tuxiang_test_stack.npy')
tuxiang_test_glove = np.load(PATH + 'data/maga_clas/tuxiang_test_glove.npy')
tuxiang_test_answ = np.load(PATH + 'data/maga_clas/tuxiang_test_answ.npy')

in_im = Input(shape=(7616,), name='1')
im = Dense(DIM, activation='relu', kernel_initializer=HE, name='2')(in_im)
me = Dense(35, activation='softmax', name='3')(im)
EXP = 'tuxiangstack'
checkpoint = ModelCheckpoint(PATH + 'data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=3, verbose=1, min_lr=0.0000001)
stop = EarlyStopping(patience=7, verbose=1)
call_list = [checkpoint,plateau,stop]
model_tuxiangstack = Model(inputs=in_im, outputs=me)
model_tuxiangstack.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])
history = model_tuxiangstack.fit(x=tuxiang_train_stack, y=tuxiang_train_answ, batch_size=32, epochs=9999, callbacks=call_list,
                                  validation_data=(tuxiang_valid_stack,tuxiang_valid_answ), class_weight='auto')

model_tuxiangstack.load_weights(PATH + 'data/weight/tuxiangstack.hdf5', by_name=True)
count = 0
for i in range(len(tuxiang_test_answ)):
    prob = model_tuxiangstack.predict(tuxiang_test_stack[i:i+1])
    pred = np.argsort(prob)[0][-1]
    if pred == tuxiang_test_answ[i]:
        count += 1
print(count)
print(len(tuxiang_test_answ))