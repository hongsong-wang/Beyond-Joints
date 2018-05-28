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

def rand_rotate_matrix_symbol(angle=90):
    srs = T.shared_randomstreams.RandomStreams()
    # np.pi / 180 *
    agx =  (srs.uniform()*(2*angle) - angle)*np.pi/180
    agy =  (srs.uniform()*(2*angle) - angle)*np.pi/180
    # s = srs.uniform() + ss
    Rx = T.stack(1,0,0, 0,T.cos(agx), T.sin(agx), 0, -T.sin(agx),T.cos(agx)).reshape((3,3))
    Ry = T.stack(T.cos(agy), 0, -T.sin(agy),  0,1,0,  T.sin(agy), 0, T.cos(agy)).reshape((3,3))
    # Ss = T.stack(s,0,0, 0,s,0, 0,0,s).reshape((3,3))
    # value = theano.dot(Ry, theano.dot(Rx, Ss))
    value = theano.dot(Ry, Rx)
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

    def load_label_file_one(self, filename):
        lines = open(filename, 'r').readlines()
        labels = np.array([map(int, np.array(item.strip().split(',')) ) for item in lines] )
        # labels[:,0] = labels[:,0] - 1
        labels[:,0] = labels[:,0]
        return labels

    def load_skeleton_file(self, filename):
        lines = open(filename, 'r').readlines()
        num = len(lines)
        skeleton = np.array([map(float, np.array(item.strip().split()) ) for item in lines] )
        skeleton2 = skeleton[:, self._num_joints:]
        inter_action = True
        if np.sum(skeleton2) == 0 and np.max(skeleton2) == 0 and np.min(skeleton2)==0:
            inter_action = False
        return skeleton, inter_action

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
                vid_list_uqk.append(name)
                idx_per = []
            else:
                idx_per.append(idx)
        return group_list, vid_list_uqk
        
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
        
    def load_seq_lst(self, lst_name, num_seq, trn_set=False, diff=0, ovr_num=None, margin_rate=1):
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

    def multi_pertime(self, sub_mean=False, rotate=False, hid_size = 512):
        '''
        I add SpatialDropout1D(0.05) while training all skeletons
        '''
        skt_input = Input(shape=(self._param['num_seq'], self._num_joints, self._dim_point) )
        data = skt_input
        if rotate:
            assert(self._dim_point == 6)
            # data = TransformLayer()(skt_input)

        if sub_mean:
            data = Permute((1,3,2))(data)
            data2 = TimeDistributed(TimeDistributed(Dense(1, kernel_initializer=init_mean1, trainable=False)))(data)
            data2 = Lambda(lambda x:K.repeat_elements(x, self._num_joints, axis=-1), \
                    output_shape=lambda s: (s[0], s[1], s[2], s[3]*self._num_joints))(data2)
            data = Add()([data, data2] )

        data = Reshape((self._param['num_seq'], self._num_joints*self._dim_point))(data)

        out = Bidirectional(LSTM(hid_size, return_sequences=True))(data)
        out = Bidirectional(LSTM(hid_size, return_sequences=True))(out)
        out = Bidirectional(LSTM(hid_size, return_sequences=True))(out)

        out = Activation('relu')( out)
        prob = TimeDistributed(Dense(self._num_class, activation='softmax'))(out)

        model = Model(skt_input, prob)
        model.summary()
        return model

    def test_model(self):
        '''
        '''
        model = self.multi_pertime(sub_mean=True, rotate=False)

        def read_hdf5(model, fileName):
            fid=h5py.File(fileName,'r')
            weight = []
            for i in range(len(fid.keys())):
                weight.append(fid['weight'+str(i)][:])
            model.set_weights(weight)
            
        X, Y, vid_list, start_list = self.load_seq_lst(self._param['val_file'], self._param['num_seq'],diff=2 )
        start_list = np.array(start_list)
        group_list, vid_list_uqk = self.merge_list(vid_list)

        # if 1:
        # for ipoch in range(64, 122, 4):
        for ipoch in range(80, 195, 4):
        # for ipoch in range(40, 125, 4):
            # init_file = self._param['initial_file']
            # init_file = '/home/wanghongsong/data/save_param_temp/view_pku_detect_joint_epoch%d.h5' % ipoch
            # init_file = '/home/wanghongsong/data/save_param_temp/view_pku_detect2_epoch%d.h5' % ipoch
            init_file = '/home/wanghongsong/data/save_param_temp/subj_cross_epoch%d.h5' % ipoch
            # init_file = '/home/wanghongsong/data/save_param_temp/view_new_detect_joint2_epoch%d.h5' % ipoch
            print ipoch
            
            read_hdf5(model, init_file )
            prob_val = model.predict(X, batch_size=self._param['batchsize'], verbose=0)
           
            gt_dict = {}
            res_dict = {}

            for idx, vid_name in zip(group_list, vid_list_uqk):
                vid_name = vid_name.split('_')[0]
                prob_seq = np.zeros((np.max(start_list[idx]) + self._param['num_seq'], self._num_class))
                for idx2 in idx:
                    prob_seq[start_list[idx2]:start_list[idx2]+self._param['num_seq']] = \
                        prob_seq[start_list[idx2]:start_list[idx2]+self._param['num_seq']] + prob_val[idx2]
                prob_seq = normalize(prob_seq, axis=1, norm='l1')
                # print vid_name, prob_seq.shape

                pred_labels = get_interval_frm_frame_predict(prob_seq) 
                # notice: for ground truth, action labels starts from 0, but for our detection, 0 indicates empty action
                pred_labels[:,0] = pred_labels[:,0] - 1
                # pred_labels = post_process_prediction(pred_labels, small_win=15)

                labels = self.load_label_file_one(os.path.join(self._param['label_path'], vid_name ) )
                gt_dict[vid_name] = labels
                res_dict[vid_name] = pred_labels
                
                if self._param['save_result']:
                    if not os.path.exists(self._param['save_predict_fold']):
                        os.makedirs(self._param['save_predict_fold'])
                    fid_out = open(os.path.join(self._param['save_predict_fold'], vid_name ), 'w')
                    # Note that the frame_id starts from 1, but for training data, it starts from 0
                    pred_labels[:,0] = pred_labels[:,0] + 1
                    pred_labels[:,1] = pred_labels[:,1] + 1
                    pred_labels[:,2] = pred_labels[:,2] + 1
                    for kk in xrange(pred_labels.shape[0]):
                        cmd_str = str(int(pred_labels[kk,0])) + ',' + str(int(pred_labels[kk,1])) + ',' + \
                            str(int(pred_labels[kk,2])) + ',' + str(pred_labels[kk,3])
                        fid_out.write(cmd_str + '\n')
                    fid_out.close()

            metrics = eval_detect_mAP(gt_dict, res_dict, minoverlap=0.5)
            print metrics['map']
            
def copy_val_gt():
    label_path = '/home/wanghongsong/data/action/skeleton/PKU/Train_Label_PKU_final/'
    file_set = os.listdir('data/val_joint')
    import shutil
    for fn in file_set:
        shutil.copy(label_path + fn, 'data/val_gt/')
            
def run_model():
    param = {}
    param['batchsize'] = 256
    param['num_seq'] = 200
    param['label_path'] = '/home/wanghongsong/data/action/skeleton/PKU/Train_Label_PKU_final/'
    
    param['save_result'] = True
    param['save_predict_fold'] = 'data/pred_val/'
    # param['save_predict_fold'] = 'data/val_cross/'
    
    if 0:
        param['val_file'] = '/home/wanghongsong/tools/srnn/icme/data/crs_view_val_skt'
        # model file
        param['initial_file'] = '/home/wanghongsong/data/save_param_temp/view_pku_detect2_epoch176.h5'
        param['initial_file'] = '/home/wanghongsong/data/save_param_temp/view_pku_detect_joint_epoch88.h5'
    else:
        param['val_file'] = '/home/wanghongsong/tools/srnn/icme/data/crs_subj_val_skt'
        # model file
        param['initial_file'] = '/home/wanghongsong/data/save_param_temp/pku_detect_edge_epoch188.h5'
        # param['initial_file'] = '/home/wanghongsong/data/save_param_temp/pku_detect2_epoch108.h5'
        param['initial_file'] = 'model_save/subj_joint.h5'
        param['initial_file'] = 'model_save/subj_edge.h5'
        param['initial_file'] = 'model_save/subj_cross.h5'

    model = construct_model(param)
    model.test_model()

if __name__ == '__main__':
    run_model()

