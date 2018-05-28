import os
import numpy as np
import random
import h5py

class pku_dataset(object):
    '''
    The dataset includes interaction actions and as well as individual actions
    '''
    def __init__(self, data_path, num_joints=25):
        self._data_path = data_path
        self._num_joints = num_joints
        self._file_set = self.get_all_filename()
        
    def load_label_file(self, filename):
        lines = open(os.path.join(self._data_path, 'Train_Label_PKU_final', filename), 'r').readlines()
        labels = np.array([map(int, np.array(item.strip().split(',')) ) for item in lines] )
        y = labels[:,0] - 1 # action id starts from 0
        s1 = labels[:,1] # notice: frame start index starts from 0
        s2 = labels[:,2]
        return y, s1, s2
        
    def load_label_file_one(self, filename):
        lines = open(os.path.join(self._data_path, 'Train_Label_PKU_final', filename), 'r').readlines()
        labels = np.array([map(int, np.array(item.strip().split(',')) ) for item in lines] )
        labels[:,0] = labels[:,0] - 1
        return labels
        
    def check_label_file(self):
        # labels, end frame, not included, max value can equal with number of frames
        fold = 'Train_Label_PKU_final'
        files = os.listdir(os.path.join(self._data_path, fold) )
        rate = 0
        count = 0
        conf_set = set([])
        conf_count = [0, 0, 0] # [0, 1, 2] three elements for confidence values
        label_set = set([])
        for filename in files:
            labels = self.load_label_file_one(filename)
            conf_set = conf_set | set(labels[:,-1])
            label_set = label_set | set(labels[:,0])
            conf = list(labels[:,-1])
            conf_count[0] = conf_count[0] + conf.count(0)
            conf_count[1] = conf_count[1] + conf.count(1)
            conf_count[2] = conf_count[2] + conf.count(2)
            
            if np.min(labels[:,1]) <=1:
                print np.min(labels[:,1])
            num_frm = self.get_num_frame(filename)
            if np.max(labels[:,2]) >=(num_frm-1):
                print np.max(labels[:,2]), num_frm
            
            if 0:
                skeleton, inter_action = self.load_skeleton_file(filename)
                action, start_pos, end_pos = self.load_label_file(filename)
                rate = rate + np.sum(end_pos-start_pos)*1.0/skeleton.shape[0]
                count = count + 1
                print rate/count
            # print np.sum(end_pos-start_pos)*1.0/skeleton.shape[0], np.sum(end_pos-start_pos), skeleton.shape[0]
        print conf_set
        print conf_count
        print label_set
            
    def get_all_filename(self):
        files = os.listdir(os.path.join(self._data_path, 'Train_Label_PKU_final') )
        files2 = os.listdir(os.path.join(self._data_path, 'PKU_Skeleton_Renew') )
        assert(len(files) == len(files2) )
        assert(set(files) == set(files2))
        return files
        # return [item[0:-4] for item in files]
        
    def load_skeleton_file(self, filename):
        lines = open(os.path.join(self._data_path, 'PKU_Skeleton_Renew', filename), 'r').readlines()
        num = len(lines)
        skeleton = np.array([map(float, np.array(item.strip().split()) ) for item in lines] )
        skeleton2 = skeleton[:, self._num_joints:]
        inter_action = True
        if np.sum(skeleton2) == 0 and np.max(skeleton2) == 0 and np.min(skeleton2)==0:
            inter_action = False
        return skeleton, inter_action
        
    def get_num_frame(self, filename):
        lines = open(os.path.join(self._data_path, 'PKU_Skeleton_Renew', filename), 'r').readlines()
        num = len(lines)
        return num
        
    def load_skeleton_all_save_hdf5(self, save_name):
        with open(save_name + '.txt', 'w') as fid_txt:
            fid_h5 = h5py.File(save_name + '.h5', 'w')
            for filename in self._file_set:
                skeleton, inter_action = self.load_skeleton_file(filename)
                action, start_pos, end_pos = self.load_label_file(filename)
                for idx in xrange(len(action)):
                    if start_pos[idx] < end_pos[idx]:
                        skt = skeleton[start_pos[idx]:end_pos[idx], 0:75]
                    else:
                        skt = skeleton[end_pos[idx]:start_pos[idx], 0:75]
                    # print skt.shape, action[idx]
                    keyname = filename[0:-4]+'_' + str(start_pos[idx])+'_' +str(end_pos[idx]) + '_first'
                    fid_txt.write( keyname +   ' ' + str(action[idx]) + '\n')
                    fid_h5.create_dataset(keyname, data=skt)
                    if inter_action:
                        skt = skeleton[start_pos[idx]:end_pos[idx], 75:]
                        keyname = filename[0:-4]+'_' + str(start_pos[idx])+'_' +str(end_pos[idx]) + '_second'
                        fid_txt.write( keyname + ' ' + str(action[idx]) + '\n')
                        fid_h5.create_dataset(keyname, data=skt)
            fid_h5.close()
            
    def split_cross_subject(self):
        lines = open(os.path.join(self._data_path, 'cross-subject.txt'), 'r').readlines()
        trn_lst = lines[1].strip().split(',')
        val_lst = lines[3].strip().split(',')
        val_lst = [(item.strip() + '.txt') for item in val_lst if item !='']
        trn_lst = [(item.strip()  + '.txt')  for item in trn_lst if item !='']
        return trn_lst, val_lst
        
    def split_cross_view(self):
        lines = open(os.path.join(self._data_path, 'cross-view.txt'), 'r').readlines()
        trn_lst = lines[1].strip().split(',')
        val_lst = lines[3].strip().split(',')
        val_lst = [(item.strip() + '.txt') for item in val_lst if item !='']
        trn_lst = [(item.strip() + '.txt') for item in trn_lst if item !='']
        return trn_lst, val_lst
        
    def load_sequence(self, name_lst, save_name=None):
        if save_name != None:
            fid_h5 = h5py.File(save_name + '.h5', 'w')
            fid_txt = open(save_name + '.txt', 'w')
        for filename in name_lst:
            skeleton, inter_action = self.load_skeleton_file(filename)
            labels = self.load_label_file_one(filename)
            if save_name != None:
                keyname1 = filename + '_' + str(inter_action)
                keyname2 = filename + '_labels'
                fid_txt.write( keyname1 + ' ' + keyname2 + '\n')
                fid_h5.create_dataset(keyname1, data=skeleton)
                fid_h5.create_dataset(keyname2, data=labels)
            
if __name__ == '__main__':
    data_path = '/home/wanghongsong/data/PKU/'
    db = pku_dataset(data_path)
    # all_lst = db.get_all_filename()
    trn_lst, val_lst = db.split_cross_view()
    db.load_sequence(trn_lst, 'data/crs_view_trn_skt')
    db.load_sequence(val_lst, 'data/crs_view_val_skt')
    # db.get_all_filename()
    # db.load_skeleton_all_save_hdf5('data/pku_clip_skt')
    # db.load_skeleton_file('PKU_Skeleton_Renew/0002-L.txt')
