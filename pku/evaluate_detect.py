import numpy as np
import os

def VOCap(rec, prec):
    '''
        Compute the average precision following the code in Pascal VOC toolkit
    '''
    mrec = np.array(rec).astype(np.float32)
    mprec = np.array(prec).astype(np.float32)
    mrec = np.insert(mrec, [0, mrec.shape[0]], [0.0, 1.0])
    mprec = np.insert(mprec, [0, mprec.shape[0]], [0.0, 0.0])

    for i in range(mprec.shape[0]-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i = np.ndarray.flatten(np.array(np.where(mrec[1:] != mrec[0:-1]))) + 1
    ap = np.sum(np.dot(mrec[i] - mrec[i-1], mprec[i]))
    return ap

def iou_ratio(bbox_1, bbox_2):
    '''
        Compute the IoU ratio between two bounding boxes
    '''
    bi = [max(bbox_1[0], bbox_2[0]), min(bbox_1[1], bbox_2[1])]
    iw = bi[1] - bi[0] + 1
    ov = 0
    if iw > 0:
        ua = (bbox_1[1] - bbox_1[0] + 1) + (bbox_2[1] - bbox_2[0] + 1) - iw
        ov = iw / float(ua)
    return ov

def compute_metric_class(gt, res, cls, minoverlap):
    npos = 0
    gt_cls = {}
    for img in gt.keys():
        index = np.array(gt[img]['class']) == cls
        BB = np.array(gt[img]['bbox'])[index]
        det = np.zeros(np.sum(index[:]))
        npos += np.sum(index[:])
        gt_cls[img] = {'BB': BB,
                       'det': det}

    # loading the detection result
    score = np.array(res[cls]['score'])
    imgs = np.array(res[cls]['img'])
    BB = np.array(res[cls]['bbox'])

    # sort detections by decreasing confidence
    si = np.argsort(-score)
    imgs = imgs[si]
    if len(BB) > 0:
        BB = BB[si, :]
    else:
        BB = BB

    # assign detections to ground truth objects
    nd = len(score)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        img = imgs[d]
        if len(BB) > 0:
            bb = BB[d, :]
        else:
            bb = BB

        ovmax = 0
        for j in range(len(gt_cls[img]['BB'])):
            bbgt = gt_cls[img]['BB'][j]
            ov = iou_ratio(bb, bbgt)
            if ov > ovmax:
                ovmax = ov
                jmax = j

        if ovmax >= minoverlap:
            if not gt_cls[img]['det'][jmax]:
                tp[d] = 1
                gt_cls[img]['det'][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp/(fp+tp)

    ap = VOCap(rec, prec)

    return rec, prec, ap


def get_action_set(gt_dict):
    gt = {}
    classes = set()
    for name in gt_dict.keys():
        labels = gt_dict[name]
        gt[name] = {}
        gt[name]['class'] = []
        gt[name]['bbox'] = []
        for idx in xrange(labels.shape[0]):
            gt[name]['class'].append(labels[idx,0])
            gt[name]['bbox'].append([labels[idx,1], labels[idx,2] ])
        classes = classes | set(gt[name]['class'] )
    return classes

def eval_detect_mAP(gt_dict, res_dict, minoverlap=0.5):
    '''
    gt_dict, the key value is a sequence name
    '''
    gt = {}
    classes = set()
    for name in gt_dict.keys():
        labels = gt_dict[name]
        gt[name] = {}
        gt[name]['class'] = []
        gt[name]['bbox'] = []
        for idx in xrange(labels.shape[0]):
            gt[name]['class'].append(labels[idx,0])
            gt[name]['bbox'].append([labels[idx,1], labels[idx,2] ])
        classes = classes | set(gt[name]['class'] )
    
    res = {}
    for cls in classes:
        res[cls] = {}
        res[cls]['img'] = []
        res[cls]['score'] = []
        res[cls]['bbox'] = []
    for name in res_dict.keys():
        pred_labels = res_dict[name]
        for idx in xrange(pred_labels.shape[0]):
            cls = int(pred_labels[idx,0])
            res[cls]['img'].append(name)
            res[cls]['score'].append(pred_labels[idx,3])
            res[cls]['bbox'].append([pred_labels[idx,1], pred_labels[idx,2] ])
    
    metrics = {}
    res_map = []
    # res_f1 = []
    for cls in classes:
        rec, prec, ap = compute_metric_class(gt, res, cls, minoverlap)
        metrics[cls] = {}
        metrics[cls]['recall'] = rec
        metrics[cls]['precision'] = prec
        metrics[cls]['ap'] = ap
        res_map.append(ap)
        
        # f1 = 2*rec*prec/(rec + prec)
        # f1[np.isnan(f1)] = 0
        # res_f1.append( np.mean(f1) )

    metrics['map'] = np.mean(np.array(res_map))
    # metrics['f1'] = np.mean(np.array(res_f1))
    return metrics

def get_interval_frm_frame_predict(prob_seq,  win_lg=15, win_sm=2, rate_min=0.1):
    # labels is 2d array, each row, (action, start_frame, end_frame)
    # prob_seq is probability predicted for each frame, (num_frame, num_class)
    # class index 0 represent empty action
    pred_seq = np.argmax(prob_seq, axis=1)
    # print pred_seq
    # accumulative count for each class and for each frame
    cur_count = np.zeros(prob_seq.shape)
    for idx in xrange(len(pred_seq)):
        if idx==0:
            cur_count[idx, pred_seq[idx]] = 1
        else:
            cur_count[idx] = cur_count[idx-1]
            cur_count[idx, pred_seq[idx]] = cur_count[idx, pred_seq[idx]] + 1
    # search for windows
    pred_labels = []
    idx = 0
    while idx < (len(pred_seq) - win_lg):
        # assume start frame idx
        if pred_seq[idx] > 0:
            rate = (cur_count[idx+win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/win_lg
            if rate > rate_min and (idx + 2*win_lg) < cur_count.shape[0]:
                # find predictions
                pos = idx + win_lg
                # search from right at window size of win_lg
                # new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/(win_lg + pos - idx)
                new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[pos, pred_seq[idx] ])*1.0/win_lg
                # print pos, new_rate, cur_count[pos+win_lg, pred_seq[idx] ], cur_count[idx, pred_seq[idx] ]
                while new_rate > rate_min and (pos + 2*win_lg) < cur_count.shape[0]:
                    pos = pos + win_lg
                    # new_rate = (cur_count[pos + win_lg, pred_seq[idx] ] - cur_count[idx, pred_seq[idx] ])*1.0/(win_lg + pos - idx)
                    new_rate = (cur_count[pos+win_lg, pred_seq[idx] ] - cur_count[pos, pred_seq[idx] ])*1.0/win_lg
                    # print new_rate
                # search from left at window size of 1
                if 0:
                    while (pred_seq[pos] != pred_seq[idx]) and pos>idx:
                        pos = pos - win_sm
                else:
                    new_rate = (cur_count[pos, pred_seq[idx] ] - cur_count[pos - win_sm, pred_seq[idx] ])*1.0/win_sm
                    while new_rate < rate_min and pos>idx:
                        pos = pos - win_sm
                        new_rate = (cur_count[pos, pred_seq[idx] ] - cur_count[pos - win_sm, pred_seq[idx] ])*1.0/win_sm
                        # print new_rate
                # assert(idx < pos)
                if idx < pos:
                    conf = np.average(prob_seq[idx:(pos+1), pred_seq[idx]] )
                    conf = round(conf, 5)
                    pred_labels.append([pred_seq[idx], idx, pos, conf ])
                    idx = pos
                else:
                    idx = idx + 1
            else:
                idx = idx + 1
        else:
            idx = idx + 1
    pred_labels = np.array(pred_labels)
    return pred_labels
    
def post_process_prediction(pred_labels, small_win=30):
    '''
    postporcessing operation for prediction
    '''
    num_seq = pred_labels.shape[0]
    remove_list = []
    for idx in xrange(num_seq):
        seg_len = pred_labels[idx,2] - pred_labels[idx,1]
        if seg_len < small_win:
            if idx==0:
                pred_labels[idx,0] = pred_labels[idx+1,0]
            if idx==num_seq-1:
                pred_labels[idx,0] = pred_labels[idx-1,0]
            if idx > 0 and idx < (num_seq-1):
                if pred_labels[idx-1,0] == pred_labels[idx+1,0]:
                    pred_labels[idx,0] = pred_labels[idx+1,0]
                else:
                    remove_list.append(idx)
    if len(remove_list):
        pred_labels = np.delete(pred_labels, np.array(remove_list), axis=0)
        pass
    return pred_labels
