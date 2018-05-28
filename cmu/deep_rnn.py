import numpy as np
import scipy
import os
from scipy import linalg
import random
import math
import h5py
import theano
import keras
from theano import tensor as T
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU, Bidirectional, Merge, \
    SimpleRNN, Input, SpatialDropout1D, Convolution1D, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.merge import Add, Concatenate, Maximum, Average
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

from cmu_mocap import cmu_mocap

def init_mean1(shape, dtype=None, name=None):
    value = np.array([-1.0/5,0,0,0,0,  0,0,-1.0/5,0,0,  0,0,0,0,0,  0,0,-1.0/5,-1.0/5,-1.0/5,  0,0,0,0,0,  0,0,0,0,0,  0 ])
    value = np.reshape(value, shape)
    # return K.random_normal(shape, dtype=dtype)
    return value

def init_mean2(shape, dtype=None, name=None):
    value = np.array([0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,-1.0/4,  -1.0/4,0,0,0,0,  0,-1.0/4,0,0,0,  0,-1.0/4,0,0,0,  0 ])
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

def rand_coordinate_symbol(rand_prob=0.5, dxx=0.6, dxy=0.6, dxz=0.1, dzx=0.02, dzy=0.02, dzz=0.02):
    if random.random() >= rand_prob:
        # x = np.array([random.uniform(-dxx, dxx)+1, random.uniform(-dxy, dxy),   random.uniform(-dxz, dxz)])
        x = np.array([random.uniform(-dxx, dxx), random.uniform(-dxy, dxy),   random.uniform(-dxz, dxz)])
        x = x/linalg.norm(x)
        z1 = np.array([random.uniform(-dzx, dzx), random.uniform(-dzy, dzy),   1 + random.uniform(-dzz, dzz)])
        y = np.cross(z1, x)
        y = y/linalg.norm(y)
        z = np.cross(x, y)
        z = z/linalg.norm(z)
        value = np.array([x, y, z], dtype=np.float32)
    else:
        value = np.eye(3, dtype=np.float32)
    return theano.shared(value.T)

class TransformLayer(Layer):
    def __init__(self, **kwargs):
        super(TransformLayer, self).__init__(**kwargs)

    # return K.in_train_phase(K.dot(x, rand_coordinate_symbol() ), x, training=training)
    def call(self, x, training=None):
        return K.in_train_phase(K.dot(x, rand_rotate_matrix_symbol()), x, training=training)
        # return K.in_train_phase(T.concatenate([K.dot(x[:,:,:,0:3], rand_rotate_matrix_symbol()), x[:,:,:,3:6] ], axis=3), x, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

class construct_model(object):
    def __init__(self, param, dim_point=3, num_joints=31, num_class=65):
        self._param = param
        self._dim_point = dim_point
        self._num_joints = num_joints
        self._num_class = num_class

    def base_model(self, rotate=False, sub_mean=False, full_rnn=False):
        '''
        use stacked two layers as baseline, use stacked three layers later
        '''
        skt_input = Input(shape=(self._param['num_seq'], self._num_joints, self._dim_point) ) # To fix length of sequence
        data = skt_input
        if rotate:
            # assert(self._dim_point == 3)
            data = TransformLayer()(data)

        if sub_mean:
            data = Permute((1,3,2))(data)
            data2 = Dense(1, kernel_initializer=init_mean1, trainable=False)(data)
            data2 = Lambda(lambda x:K.repeat_elements(x, self._num_joints, axis=-1), \
                output_shape=lambda s: (s[0], s[1], s[2], s[3]*self._num_joints))(data2)
            # data = Merge(mode='sum')([data, data2])
            data = Add()([data, data2] )
            data = Reshape((self._param['num_seq'], self._num_joints*self._dim_point))(data)
        else:
            data = Reshape((self._param['num_seq'], self._num_joints*self._dim_point))(data)

        data = SpatialDropout1D(0.05)(data)
        out = Bidirectional(LSTM(512, return_sequences=True))(data)
        # out = SpatialDropout1D(0.05)(out)
        out = Bidirectional(LSTM(512, return_sequences=True))(out)
        # out = SpatialDropout1D(0.05)(out)
        out = Bidirectional(LSTM(512, return_sequences=True))(out)

        if full_rnn:
            # LSTM, GRU, SimpleRNN
            prob = Bidirectional(SimpleRNN(self._num_class, return_sequences=True), merge_mode='ave' )(out)
            prob = Lambda(lambda x:T.max(x, axis=1), output_shape=lambda s: (s[0], s[2]))(prob)
            prob = Activation('softmax')(prob)
        else:
            out = Lambda(lambda x:T.max(x, axis=1), output_shape=lambda s: (s[0], s[2]))(out)
            # out = BatchNormalization()(out)
            out = Dropout(0.5)(out)
            out = Activation('relu')(out)
            prob = Dense(self._num_class, activation='softmax')(out)

        model = Model(skt_input, prob)
        opt = SGD(lr=self._param['base_learn_rate'], decay=self._param['weight_regular'], momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def group_person_list(self, list_file):
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        vdname_list = [line[0:line.index('.skeleton')] for line in name_list ]
        label_list = [(int(name[17:20])-1) for name in name_list]
        idx_per = []
        group_list = []
        for idx, name in enumerate(name_list):
            vdname = vdname_list[idx]
            if idx == len(name_list)-1:
                last_vdname = ''
            else:
                last_vdname = vdname_list[idx+1]
            if vdname != last_vdname:
                idx_per.append(idx)
                # there exist samples with 3 skeletons, to check standard deviation
                # print len(idx_per), idx_per
                group_list.append(idx_per)
                # print [label_list[temp] for temp in idx_per]
                idx_per = []
            else:
                idx_per.append(idx)
        return group_list

    def train_model(self):
        model = self.base_model(rotate=False, sub_mean=False, full_rnn=False)

        db = cmu_mocap(self._param['data_path'])
        if self._param['subset']:
            trn_lst, val_lst = db.subset_cross_validation(self._param['split'])
        else:
            trn_lst, val_lst = db.cross_validation(self._param['split'])
        trainX, trainY, train_vid_list = db.load_skeleton_lst(trn_lst, num_seq= self._param['num_seq'],
            step=self._param['step'])
        valX, valY, val_vid_list = db.load_skeleton_lst(val_lst, num_seq= self._param['num_seq'],
            step=self._param['step'])

        trainY = np_utils.to_categorical(trainY, self._num_class )
        valY = np_utils.to_categorical(valY, self._num_class )

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
        if self._param['initial_file'] is not None:
            read_hdf5(model, self._param['initial_file'] )

        class evaluateVal(keras.callbacks.Callback):
            def __init__(self, vid_list):
                self.group_list, self.gt_val = self.merge_list(vid_list)

            def merge_list(self, vid_list):
                group_list = []
                gt_val = []
                idx_per = []
                for idx, name in enumerate(vid_list):
                    if idx == len(vid_list)-1:
                        last_name = ''
                    else:
                        last_name = vid_list[idx+1]
                    if name != last_name:
                        idx_per.append(idx)
                        gt_val.append(np.argmax(valY[idx]) )
                        group_list.append(np.asarray(idx_per) )
                        idx_per = []
                    else:
                        idx_per.append(idx)
                return group_list, gt_val

            def on_epoch_end(self, epoch, logs={}):
                if (epoch % 2==0):
                    val_loss = model.evaluate(valX, valY, batch_size=512, verbose=0)[0]
                    prob_val = model.predict(valX, batch_size=512, verbose=0)
                    pred = np.asarray([np.argmax(np.mean(prob_val[idx], axis=0)) for idx in self.group_list ] )
                    acc = sum( int(pred[i]) == self.gt_val[i] for i in xrange(len(self.gt_val))) / float(len(self.gt_val))
                    train_loss = model.evaluate(trainX, trainY, batch_size=512, verbose=0)[0]
                    cmd_str = 'evluation epoch=%d, learn_rate=%f, train loss=%f, validation loss=%f, validation accuracy=%f' % (epoch,
                        K.get_value(model.optimizer.lr), train_loss, val_loss, acc)
                    print cmd_str
                    # if 'fid_out' in locals() or 'fid_out' in globals():
                    if write_file:
                        fid_out.write(cmd_str + '\n')
                    if (epoch % 8==0) and epoch > 0 and save_model:
                        save_file = save_path + ('_epoch%d.h5' % epoch)
                        if os.path.exists(save_file):
                            os.remove(save_file)
                        # model.save_weights(save_file)
                        save_hdf5(model, save_file)

        reduce_lr = LearningRateScheduler(schedule)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.001)

        evaluate_val = evaluateVal(val_vid_list)

        model.fit(trainX, trainY, batch_size=self._param['batchsize'], epochs=self._param['max_iter'],
            callbacks=[evaluate_val, reduce_lr ], shuffle=True, verbose=0 )

def run_model():
    param = {}
    param['max_iter'] = 2000
    # notice: changed learning rate and step size
    param['step_inter'] = 60
    param['base_learn_rate'] = 0.01 #  defaults 0.02
    param['lr_gamma'] = 0.7 # default 0.1
    param['weight_regular'] = 0
    param['batchsize'] = 256 # 64

    param['data_path'] = '/home/wanghongsong/data/cmu/'
    param['subset'] = False
    param['split'] = 1
    param['num_seq'] = 100
    param['step'] = 1

    param['write_file'] = True
    param['write_file_name'] = 'deep_view2.txt'
    param['save_model'] = True
    param['save_path'] = '/home/wanghongsong/data/save_param_temp/cmu_deep'
    param['initial_file'] = None

    param['rand_start'] = True
    param['max_start_rate'] = 0.8
    param['rand_view'] = True

    model = construct_model(param)
    model.train_model()

if __name__ == '__main__':
    run_model()
