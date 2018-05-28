import os
import string
import random
import math
import numpy as np
from scipy import linalg

def PCA(data, dims_rescaled_data=2):
    m, n = data.shape
    mean_coord = data.mean(axis=0)
    new_data = data - mean_coord
    # calculate the covariance matrix
    R = np.cov(new_data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # new_data = np.dot(evecs.T, data.T).T
    # new_data = np.dot(new_data, evecs)
    # print new_data.shape
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return evecs, mean_coord

class cmu_mocap(object):
    def __init__(self, data_path, classNum=45, num_joints=31):
        self._data_path = data_path
        self._classNum = classNum
        self._num_joints = num_joints
        name_lst, label = self.get_filename_label()
        self.name2label = dict(zip(name_lst, label) )

    def get_filename_label(self):
        lines = open(os.path.join(self._data_path, 'skeletons', 'label.txt'), 'r').readlines()
        label = [int(item.strip())-1 for item in lines] # starts from 0
        lines = open(os.path.join(self._data_path, 'skeletons', 'filelist.txt'), 'r').readlines()
        name_lst = [item.strip() for item in lines]
        assert(len(name_lst) == len(label) )
        return name_lst, label

    def load_skeleton_file(self, filename):
        lines = open(os.path.join(self._data_path, 'skeletons', filename), 'r').readlines()
        num = len(lines)
        skeleton = np.array([map(float, np.array(item.strip().split(',')) ) for item in lines] )
        skeleton = skeleton.reshape((skeleton.shape[0], self._num_joints, 3))
        return skeleton

    def spatial_diff(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        fidx = [19,1,2,3,4,5,6,  19,8,9,10,11,12,13,  15,15,16,17,18,19,20,  15,22,23,24,25,  15,27,28,29,30]
        assert(len(fidx) == skeleton.shape[1] )
        return skeleton[:,np.array(fidx)-1 ] - skeleton
        # return np.concatenate((skeleton, skeleton[:,np.array(fidx)-1 ] - skeleton ), axis=2)

    def spatial_cross(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        fidx1 = [19,1,2,3,4,5,5,  19,8,9,10,11,12,12,   22,15,16,17,1,18,19,  15,22,23,24,24,  15,27,28,29,29]
        fidx2 = [2,3,4,5,6,7,6,   9,10,11,12,13,14,13,  27,17,18,19,8,19,20,  23,24,25,26,25,  28,29,30,31,30]
        skt1 = skeleton[:,np.array(fidx1)-1 ] - skeleton
        skt2 = skeleton[:,np.array(fidx2)-1 ] - skeleton
        return np.cross(skt1, skt2)

    def load_skeleton_lst(self, name_lst, num_seq=100, ovr_num=None, step=1, spatil_diff=True):
        '''
        # adjust_corrd=0
        skt_shape = skeleton_raw.shape
        skt_per = skeleton_raw.reshape((-1, 3))
        evecs, mean_coord = PCA(skt_per)
        z_drt = evecs[:,0]/linalg.norm(evecs[:,0])
        x_drt = evecs[:,1]/linalg.norm(evecs[:,1])
        y_drt = np.cross(z_drt, x_drt)
        y_drt = y_drt/linalg.norm(y_drt)
        x_drt = np.cross(y_drt, z_drt)
        x_drt = x_drt/linalg.norm(x_drt)
        adjust = np.array([x_drt, y_drt, z_drt], dtype=np.float32)
        skt_per = np.dot(skt_per-mean_coord, adjust.T ) # subtract center
        # skt_per = np.dot(skt_per, adjust.T )
        skeleton_raw = skt_per.reshape(skt_shape)
        '''
        if ovr_num == None:
            ovr_num = num_seq/2
        X = []
        Y = []
        vid_list = []
        for name in name_lst:
            label_per = self.name2label[name]
            skeleton_raw = self.load_skeleton_file(name)

            if spatil_diff:
                skeleton_raw = self.spatial_diff(skeleton_raw)
                # skeleton_raw = self.spatial_cross(skeleton_raw)

            for step_idx in xrange(step):
                skeleton = skeleton_raw[step_idx::step]
                if skeleton.shape[0] > num_seq:
                    start = 0
                    while start + num_seq < skeleton.shape[0]:
                        X.append(skeleton[start:start+num_seq])
                        Y.append(label_per)
                        vid_list.append(name)
                        start = start + ovr_num
                    X.append(skeleton[-num_seq:])
                    Y.append(label_per)
                    vid_list.append(name)
                else:
                    skeleton = np.concatenate((np.zeros((num_seq-skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])), skeleton), axis=0)
                    X.append(skeleton)
                    Y.append(label_per)
                    vid_list.append(name)
        X = np.asarray(X).astype(np.float32)
        Y = (np.asarray(Y)).astype(np.int32)
        return X, Y, vid_list

    def subset_cross_validation(self, idx=1):
        assert(idx in [1, 2, 3]), 'split index must belongs to {1, 2, 3} '
        name_lst1 = [item.strip() for item in open(os.path.join(self._data_path, 'split_test', 'filelist1.txt'), 'r').readlines()]
        name_lst2 = [item.strip() for item in open(os.path.join(self._data_path, 'split_test', 'filelist2.txt'), 'r').readlines()]
        name_lst3 = [item.strip() for item in open(os.path.join(self._data_path, 'split_test', 'filelist3.txt'), 'r').readlines()]
        if idx==1:
            trn_lst = name_lst2 + name_lst3
            val_lst = name_lst1
        if idx==2:
            trn_lst = name_lst1 + name_lst3
            val_lst = name_lst2
        if idx==3:
            trn_lst = name_lst1 + name_lst2
            val_lst = name_lst3
        return trn_lst, val_lst

    def cross_validation(self, idx=1):
        assert(idx in [1, 2, 3, 4]), 'split index must belongs to {1, 2, 3, 4} '
        trn_lst = [item.strip() for item in open(os.path.join(self._data_path, 'split_test', 'trainfilelistfold%d.txt'% idx), 'r').readlines()]
        val_lst = [item.strip() for item in open(os.path.join(self._data_path, 'split_test', 'testfilelistfold%d.txt'% idx), 'r').readlines()]
        return trn_lst, val_lst

    def check_sequence(self):
        name_lst, label = self.get_filename_label()
        for filename in name_lst:
            skeleton = self.load_skeleton_file(filename)
            print skeleton.shape

if __name__ == '__main__':
    data_path = '/home/wanghongsong/data/cmu/'
    db = cmu_mocap(data_path)
    # db.load_skeleton_file('01_01.txt')
    # db.get_filename_label()
    db.check_sequence()
