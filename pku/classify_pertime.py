import numpy as np
import os
import scipy
from scipy import linalg
from sklearn.preprocessing import normalize
import random
import math
import h5py
import theano
import keras
from theano import tensor as T
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, \
    GRU, SimpleRNN, Input, SpatialDropout1D, Reshape, Permute, Lambda, TimeDistributed
from keras.layers.merge import Add, Concatenate, Maximum
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop,SGD,Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm

from evaluate_detect import *

def init_mean1(shape, dtype=None, name=None):
    value = np.array([0,0,-1.0/4,0,-1.0/4,  0,0,0,-1.0/4,0,  0,0,0,0,0,  0,0,0,0,0,  -1.0/4,0,0,0,0 ])
    value = np.reshape(value, shape)
    return value
    
def init_mean2(shape, dtype=None, name=None):
    value = np.array([-1.0/4,-1.0/4,0,0,0,  0,0,0,0,0,  0,0,-1.0/4,0,0,  0,-1.0/4,0,0,0,  0,0,0,0,0 ])
    value = np.reshape(value, shape)
    return value
    
def rand_rotate_matrix(shape, dtype=None, name=None, angle1=-90, angle2=90, s1=0.5, s2=1.5):
    random.random()
    agx = random.uniform(angle1, angle2)  # do not use randint
    agy = random.uniform(angle1, angle2)
    s = random.uniform(s1, s2)
    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
    value = np.dot(Ry,np.dot(Rx,Ss))
    value = np.reshape(value, shape)
    return value
    
def rand_rotate_matrix_symbol(angle=90, ss=0.5):
    srs = T.shared_randomstreams.RandomStreams()
    # np.pi / 180 *
    agx =  (srs.uniform()*(2*angle) - angle)*np.pi/180
    agy =  (srs.uniform()*(2*angle) - angle)*np.pi/180
    s = srs.uniform() + ss
    Rx = T.stack(1,0,0, 0,T.cos(agx), T.sin(agx), 0, -T.sin(agx),T.cos(agx)).reshape((3,3))
    Ry = T.stack(T.cos(agy), 0, -T.sin(agy),  0,1,0,  T.sin(agy), 0, T.cos(agy)).reshape((3,3))
    Ss = T.stack(s,0,0, 0,s,0, 0,0,s).reshape((3,3))
    value = theano.dot(Ry, theano.dot(Rx, Ss))
    return value
        
class TransformLayer(Layer):
    def __init__(self, **kwargs):
        super(TransformLayer, self).__init__(**kwargs)
        
    def call(self, x, training=None):
        rr = rand_rotate_matrix_symbol()
        # return K.in_train_phase(K.dot(x, rr), x, training=training)
        return K.in_train_phase(T.concatenate([K.dot(x[:,:,:,0:3], rr), K.dot(x[:,:,:,3:], rr)], axis=-1), x, training=training)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
class construct_model(object):
    def __init__(self, param, dim_point=3*2, num_joints=25, num_class=51 ):
        self._param = param
        self._dim_point = dim_point
        self._num_joints = num_joints
        # one additional class for empty action
        self._num_class = num_class + 1
        
    def func_stacked_lstm(self, data, hid_size = 512):
        lstm1 = Bidirectional(LSTM(hid_size, return_sequences=True))
        lstm2 = Bidirectional(LSTM(hid_size, return_sequences=True))
        lstm3 = Bidirectional(LSTM(hid_size, return_sequences=True))
        
        out = lstm3(lstm2(lstm1(data)) )
        out = Lambda(lambda x:T.max(x, axis=1), output_shape=lambda s: (s[0], s[2]))(out)
        out = Activation('relu')( Dropout(0.5)(out) )
        prob = Dense(self._num_class, activation='softmax')(out)
        return prob
        
    def load_label_map(self, lst_name):
        assert(os.path.isfile(lst_name + '.txt') and os.path.isfile(lst_name + '.h5') )
        keyname_lst = [item.strip().split() for item in open(lst_name + '.txt', 'r').readlines()]
        label_map = {}
        with h5py.File(lst_name + '.h5','r') as hf:
            for item in keyname_lst:
                labels = np.asarray(hf.get(item[1]))
                # action labels for detection starts from 1
                labels[:,0] = labels[:,0] + 1
                label_map[item[1]] = labels
        return label_map
        
    def spatial_diff(self, skeleton):
        # assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        fidx = [1, 1, 21, 3,    21, 5,6,7,   21,9,10,11,   1,13,14,15,   1, 17,18,19,  2,8,8,12,12 ]
        assert(len(fidx) == skeleton.shape[1] )
        return skeleton[:,np.array(fidx)-1 ] - skeleton

    def spatial_cross(self, skeleton):
        # assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        # fidx1 = [17,21,4,21,  6,5,6,22,  21,11,12,24,  1,13,16,14,  18,17,18,19,  5,8,8,  12,12]
        # fidx2 = [13,1,21,3,  21,7,8,23,  10,9,10,25,  14,15,14,15,  1,19,20,18,  9,23,22, 25,24]
        fidx1 = [17,21,9,21,  6,5,6,22,  21,11,12,24,  2,1,13,14,  2,17,18,18,  2,8,8,  12,12]
        fidx2 = [13,1,21,3,  21,7,8,23,  10,9,10,25,  1,13,14,15,  1,1,17,19,  9,7,7, 11,11]
        skt1 = skeleton[:,np.array(fidx1)-1 ] - skeleton
        skt2 = skeleton[:,np.array(fidx2)-1 ] - skeleton
        return np.concatenate((100*np.cross(skt1[:,:,0:3], skt2[:,:,0:3]),100*np.cross(skt1[:,:,3:], skt2[:,:,3:])), axis=-1)
        
    def load_seq_lst(self, lst_name, num_seq, trn_set=True, diff=0, ovr_num=None, margin_rate=1):
        # the reverse process of load_sequence() method in pku_dataset class 
        # the output is a list of skeleton sequence of variable length
        # for training set, samples clips around intervals of actions
        assert(os.path.isfile(lst_name + '.txt') and os.path.isfile(lst_name + '.h5') )
        if ovr_num == None:
            ovr_num = num_seq/2
        keyname_lst = [item.strip().split() for item in open(lst_name + '.txt', 'r').readlines()]
        X = []
        Y = []
        vid_list = []
        start_list = []
        with h5py.File(lst_name + '.h5','r') as hf:
            for item in keyname_lst:
                skeleton = np.asarray(hf.get(item[0]))
                # skeleton = skeleton.reshape((skeleton.shape[0], self._num_joints, self._dim_point ))
                skeleton1 = skeleton[:, 0:75].reshape((skeleton.shape[0], self._num_joints, 3 ))
                skeleton2 = skeleton[:, 75:].reshape((skeleton.shape[0], self._num_joints, 3 ))
                skeleton = np.concatenate((skeleton1, skeleton2), axis=-1)
                
                if diff == 1:
                    skeleton = self.spatial_diff(skeleton)
                if diff == 2:
                    skeleton = self.spatial_cross(skeleton)
                
                labels = np.asarray(hf.get(item[1]))
                labels_pertime = np.zeros((skeleton.shape[0]), dtype=np.int32)
                for clip_idx in xrange(len(labels)):
                    # notice: for detection, labels start from 1, as there are empty clips for the input stream
                    labels_pertime[labels[clip_idx][1]:labels[clip_idx][2]] = labels[clip_idx][0] + 1
                labels_pertime = labels_pertime.astype(np.int32)
                labels_pertime = np_utils.to_categorical(labels_pertime, self._num_class )
                
                if trn_set:
                    for clip_idx in xrange(len(labels)):
                        # only sample clips centered at each action
                        if labels[clip_idx][1] > labels[clip_idx][2]:
                            temp = labels[clip_idx][2]
                            labels[clip_idx][2] = labels[clip_idx][1]
                            labels[clip_idx][1] = temp
                        pos1 = np.max([labels[clip_idx][1] - margin_rate*num_seq, 0])
                        pos2 = np.min([labels[clip_idx][2] + margin_rate*num_seq, skeleton.shape[0] ])
                        assert(skeleton.shape[0] >= labels[clip_idx][2] and pos1 < pos2)
                        skt = skeleton[pos1:pos2]
                        label_pt = labels_pertime[pos1:pos2]
                        if skt.shape[0] > num_seq:
                            start = 0
                            while start + num_seq < skt.shape[0]:
                                X.append(skt[start:start+num_seq])
                                Y.append(label_pt[start:start+num_seq])
                                vid_list.append(item[0])
                                start_list.append(start + pos1 )
                                start = start + ovr_num
                                
                            X.append(skt[-num_seq:])
                            Y.append(label_pt[-num_seq:])
                            vid_list.append(item[0])
                            start_list.append(skeleton.shape[0]-num_seq)
                        else:
                            print pos1, pos2, skt.shape
                            if pos1 - (num_seq - skt.shape[0]) > 0:
                                pos1 = pos1 - (num_seq - skt.shape[0])
                            else:
                                pos2 = pos2 + (num_seq - skt.shape[0])
                            X.append(skeleton[pos1:pos2])
                            Y.append(labels_pertime[pos1:pos2])
                            vid_list.append(item[0])
                            start_list.append(pos1)
                else:
                    if skeleton.shape[0] > num_seq:
                        start = 0
                        while start + num_seq < skeleton.shape[0]:
                            X.append(skeleton[start:start+num_seq])
                            Y.append(labels_pertime[start:start+num_seq])
                            vid_list.append(item[0])
                            start_list.append(start)
                            start = start + ovr_num
                        X.append(skeleton[-num_seq:])
                        Y.append(labels_pertime[-num_seq:])
                        vid_list.append(item[0])
                        start_list.append(skeleton.shape[0]-num_seq)
                    else:
                        skeleton = np.concatenate((np.zeros((num_seq-skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])), skeleton), axis=0)
                        labels_pertime = np.concatenate((np.zeros((num_seq-labels_pertime.shape[0], labels_pertime.shape[1])), labels_pertime), axis=0)
                        X.append(skeleton)
                        Y.append(labels_pertime)
                        vid_list.append(item[0])
                        start_list.append(0) 
                        
        X = np.asarray(X).astype(np.float32)
        Y = np.asarray(Y)
        print X.shape, Y.shape
        return X, Y, vid_list, start_list
        
    def multi_pertime(self, sub_mean=False, rotate=False):
        skt_input = Input(shape=(self._param['num_seq'], self._num_joints, self._dim_point) )
        data = skt_input
        if rotate:
            assert(self._dim_point == 6)
            data = TransformLayer()(skt_input)
        
        if sub_mean:
            data = Permute((1,3,2))(data)
            data2 = TimeDistributed(TimeDistributed(Dense(1, kernel_initializer=init_mean1, trainable=False)))(data)
            data2 = Lambda(lambda x:K.repeat_elements(x, self._num_joints, axis=-1), \
                    output_shape=lambda s: (s[0], s[1], s[2], s[3]*self._num_joints))(data2)
            data = Add()([data, data2] )
        
        data = Reshape((self._param['num_seq'], self._num_joints*self._dim_point))(data)
        
        hid_size = 512
        out = Bidirectional(LSTM(hid_size, return_sequences=True))(data)
        out = Bidirectional(LSTM(hid_size, return_sequences=True))(out)
        out = Bidirectional(LSTM(hid_size, return_sequences=True))(out)
        
        out = Activation('relu')( Dropout(0.5)(out) )
        prob = TimeDistributed(Dense(self._num_class, activation='softmax'))(out)

        opt = SGD(lr=self._param['base_learn_rate'], decay=self._param['weight_regular'], momentum=0.9, nesterov=True)
        model = Model(skt_input, prob)
        model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy']) 
        model.summary()
        return model
        
    def test_model(self):
        model = self.multi_pertime()
        valX, valY, val_vid_list, val_start_list = self.load_seq_lst(self._param['val_file'], self._param['num_seq'], trn_set=False)
        val_start_list = np.array(val_start_list)
        label_map = self.load_label_map(self._param['val_file'])
        
        def read_hdf5(model, fileName):
            fid=h5py.File(fileName,'r')
            weight = []
            for i in range(len(fid.keys())):
                weight.append(fid['weight'+str(i)][:])
            model.set_weights(weight)
            
        def merge_list(vid_list):
            group_list = []
            idx_per = []
            vid_list_uqk = []
            for idx, name in enumerate(vid_list):
                if idx == len(vid_list)-1:
                    last_name = ''
                else:
                    last_name = vid_list[idx+1]
                if name != last_name:
                    idx_per.append(idx)
                    group_list.append(np.asarray(idx_per) )
                    vid_list_uqk.append(name.split('_')[0] + '_labels')
                    idx_per = []
                else:
                    idx_per.append(idx)
            return group_list, vid_list_uqk
            
        assert(self._param['initial_file'] != None )
        read_hdf5(model, self._param['initial_file'])
        group_list, vid_list_uqk = merge_list(val_vid_list)
        
        prob_val = model.predict(valX, batch_size=self._param['batchsize'], verbose=0)
        precision, recall, f1 = 0, 0, 0
        count = 0
        for idx, vid_name in zip(group_list, vid_list_uqk):
            print vid_name
            prob_seq = np.zeros((np.max(val_start_list[idx]) + self._param['num_seq'], self._num_class))
            for idx2 in idx:
                prob_seq[val_start_list[idx2]:val_start_list[idx2]+self._param['num_seq']] = \
                    prob_seq[val_start_list[idx2]:val_start_list[idx2]+self._param['num_seq']] + prob_val[idx2]
            prob_seq = normalize(prob_seq, axis=1, norm='l1')
            labels = label_map[vid_name]
            
            pred_labels = get_interval_frm_frame_predict(prob_seq)
            # print pred_labels
            # print labels
            # to evaluate using mAP
            pass
        
    def train_model(self):
        model = self.multi_pertime(sub_mean=True, rotate=True)
        
        trainX, trainY, train_vid_list, train_start_list = self.load_seq_lst(self._param['trn_file'], self._param['num_seq'], diff=0, trn_set=True)
        valX, valY, val_vid_list, val_start_list = self.load_seq_lst(self._param['val_file'], self._param['num_seq'], diff=0, trn_set=False)
        train_start_list = np.array(train_start_list)
        val_start_list = np.array(val_start_list)
        
        label_map = self.load_label_map(self._param['val_file'])
        
        if self._param['write_file']:
            fid_out = open(self._param['write_file_name'], 'w')
        
        def save_hdf5(model, fileName):
            fid = h5py.File(fileName,'w')
            weight = model.get_weights()
            for i in range(len(weight)):
                fid.create_dataset('weight'+str(i),data=weight[i])
            fid.close()

        def read_hdf5(model, fileName):
            fid=h5py.File(fileName,'r')
            weight = []
            for i in range(len(fid.keys())):
                weight.append(fid['weight'+str(i)][:])
            model.set_weights(weight)
            
        def schedule(epoch):
            lr = K.get_value(model.optimizer.lr)
            if epoch % self._param['step_inter'] == 0 and epoch > 0:
                lr = lr*self._param['lr_gamma']
            return np.float(lr)
            
        write_file = False
        if self._param['write_file']:
            write_file = True
            fid_out = open(self._param['write_file_name'], 'w')
            
        save_model = False
        if self._param['save_model']:
            save_model = True
            save_path = self._param['save_path']
        batchsize = self._param['batchsize']
        initial_file = self._param['initial_file']
        num_seq = self._param['num_seq']
        num_class = self._num_class
        
        class evaluateVal(keras.callbacks.Callback):
            def __init__(self):
                self.group_list, self.vid_list_uqk = self.merge_list(val_vid_list)
                
            def merge_list(self, vid_list):
                group_list = []
                idx_per = []
                vid_list_uqk = []
                for idx, name in enumerate(vid_list):
                    if idx == len(vid_list)-1:
                        last_name = ''
                    else:
                        last_name = vid_list[idx+1]
                    if name != last_name:
                        idx_per.append(idx)
                        group_list.append(np.asarray(idx_per) )
                        vid_list_uqk.append(name.split('_')[0] + '_labels')
                        idx_per = []
                    else:
                        idx_per.append(idx)
                return group_list, vid_list_uqk
                
            def on_epoch_end(self, epoch, logs={}):
                if epoch == 0 and initial_file != None:
                    read_hdf5(model, initial_file)
                if epoch % 4==0 and epoch > 0:
                    save_file = save_path + ('_epoch%d.h5' % epoch)
                    save_hdf5(model, save_file)
                if epoch % 4==0 and epoch > 50:
                    gt_dict = {}
                    res_dict = {}
                    prob_val = model.predict(valX, batch_size=batchsize, verbose=0)
                    count = 0
                    for idx, vid_name in zip(self.group_list, self.vid_list_uqk):
                        prob_seq = np.zeros((np.max(val_start_list[idx]) + num_seq, num_class))
                        for idx2 in idx:
                            prob_seq[val_start_list[idx2]:val_start_list[idx2]+num_seq] = \
                                prob_seq[val_start_list[idx2]:val_start_list[idx2]+num_seq] + prob_val[idx2]
                        prob_seq = normalize(prob_seq, axis=1, norm='l1')
                        labels = label_map[vid_name]
                        pred_labels = get_interval_frm_frame_predict(prob_seq)
                        
                        gt_dict[vid_name] = labels
                        res_dict[vid_name] = pred_labels
                        
                    metrics = eval_detect_mAP(gt_dict, res_dict, minoverlap=0.5)
                    map = metrics['map']
                    cmd_str = 'epoch: %d, learn_rate=%f, map: %f' % \
                              (epoch, K.get_value(model_train.optimizer.lr), map)
                    print cmd_str
                    if write_file:
                        fid_out.write(cmd_str + '\n')
                        
        evaluate_val = evaluateVal()
        reduce_lr = LearningRateScheduler(schedule)
        
        model.fit(trainX, trainY, batch_size=self._param['batchsize'], epochs=self._param['max_iter'], shuffle=True, 
            validation_data=(valX, valY),  callbacks=[evaluate_val, reduce_lr ], verbose=1)
        
def run_model():
    param = {}
    param['max_iter'] = 500
    param['step_inter'] = 80
    param['base_learn_rate'] = 0.01 
    param['lr_gamma'] = 0.5
    param['weight_regular'] = 0
    param['num_seq'] = 200 
    param['batchsize'] = 256
    # no batch size, as one batch only 

    if 0:
        param['trn_file'] = '/home/wanghongsong/hongsong/tools/srnn/icme/data/crs_subj_train_skt'
        param['val_file'] = '/home/wanghongsong/hongsong/tools/srnn/icme/data/crs_subj_val_skt'
    else:
        param['trn_file'] = '/home/wanghongsong/hongsong/tools/srnn/icme/data/crs_view_trn_skt'
        param['val_file'] = '/home/wanghongsong/honsong/tools/srnn/icme/data/crs_view_val_skt'

    param['write_file'] = True
    param['write_file_name'] = 'subj2.txt' # 'subj.txt', 'view.txt'
    param['save_model'] = True
    param['save_path'] = '/home/wanghongsong/hongsong/data/save_param_temp/view_new_detect_joint2'
    param['initial_file'] = None
    # param['initial_file'] ='model_save/detect/view_bilstm512_512_512.h5'

    model = construct_model(param)
    model.train_model()
    # model.test_model()

        
if __name__ == '__main__':
    run_model()
    
