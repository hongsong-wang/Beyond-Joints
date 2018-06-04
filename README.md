# Beyond Joints: Learning Representations from Primitive Geometries for Skeleton-based Action Recognition and Detection
**Hongsong Wang**, Liang Wang. Beyond Joints: Learning Representations from Primitive Geometries for Skeleton-based Action Recognition and Detection. IEEE Transactions on Image Processing (**TIP 2018**), vol. 27, no. 9, pp. 4382-4394.
[[Published article]](https://ieeexplore.ieee.org/document/8360391/) 

Our codebase is based on Keras with Theano backend.

## Action Recognition on NTU RGB+D
Please read this [[ntu rgb+d repository]](https://github.com/shahroudy/NTURGB-D) and download the dataset by following this [[instruction]](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). You can only download the skeletons of this dataset.

In the **ntu_rgbd.py**, we use function **save_h5_file_skeleton_list** to save skeletons into hdf5 file. If one sample has more than one skeletons, we save one skeleton each.
During traning, we sample sub-sequences with length of 100 with an interval of 50, and assign it with the action label. For testing, we average the predicted scores of this subsequences. **If the test sample has multiple skeletons, we average scores of sub-sequences of all the skeletons**. The details are in the function **group_person_list** and the class **evaluateVal** in the **deep_rnn.py**. 

## Action Recognition on CMU mocap
We follow the experimental setup of this paper, Cooccurrence Feature Learning for Skeleton based Action Recognition using Regularized Deep LSTM Networks, AAAI 2016. The processed skeleton data can be download from the the [[website]](http://www.escience.cn/people/wentao/index.html) (search for key words: CMU subset ).  Here, we use 4-fold cross validation files for CMU dataset, not the three fold cross validation files for CMU subset. 

## Action Detection on PKU-MMD
The dataset [[PKU-MMD repository]](https://github.com/ECHO960/PKU-MMD). You can only download the skeletons. The main data processing is the function **load_seq_lst** in **classify_pertime.py**. During training, we only sample sub-squences with length of 200 around the action intervals. It seems this step is a bit important for fast training, when compared with random sampling.

## Contact 
For any question, feel free to contact:
Hongsong Wang, hongsong.wang@nlpr.ia.ac.cn
