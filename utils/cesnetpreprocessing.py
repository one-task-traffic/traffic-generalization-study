import functools

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from tensorflow.keras.utils import to_categorical

import numpy as np
import keras
import pandas as pd
import time
import pickle as pkl
import math

from . import config


'''
retrieves the classes from the adjusted dataset that have freq_thres instances or more. 
returns classnum of the above classes if classnum is not None
strict means we do not want the flows that don't have an SNI but were labeled by orange-deep-traffic heuristically.
''' 
def retrieve_popular_classes(adjusted_dataset, classnum=20, freq_thres=1000, strict=True):
    labels = [x[1][0] for x in adjusted_dataset if not strict or x[-1][0] is not None]
    cnter = Counter(labels)
    
    assert classnum==None or (classnum <= len(cnter) and classnum >=0)
    
    ignore_num = False
    ignore_freq = False
    
    if classnum ==0 or classnum == None:
        ignore_num = True
    
    if freq_thres ==0 or freq_thres == None:
        ignore_freq = True
        
    top_labels=[]
    freq_labels = []
    
    if ignore_num and ignore_freq:
        return cnter
    
    elif ignore_freq:
        top_labels = cnter.most_common()[0:classnum]
        return top_labels
    
    elif ignore_num:
        for lbl, freq in cnter.most_common():
            if freq > freq_thres:
                freq_labels.append((lbl,freq))
        return freq_labels
    
    else: 
        top_labels = [cnter.most_common()[0:classnum]]
        for lbl, freq in cnter.most_common():
            if freq > freq_thres:
                freq_labels.append((lbl,freq))
        
        if len(freq_labels)>classnum:
            return freq_labels[:classnum]
        else: return cnter.most_common()[0:classnum]
        

'''
Out of CESNET, only pick samples from the picked fine classes. 
    start_ID, end_ID: first and last index of all_labelled_IDs to look in for samples
    group_size: final sample size
    label_list: the labels for which samples should be retrieved, all samples having other labels are ignored
    label_list_level: whether label list level is fine 0 or coarse 1
    return_coarse: level of label to return in the label list
    both_label_indices: dictionary of label name to label int, needed because all_labelled_IDs contains names
'''
def generate_balanced_label_and_ID_list_for_labels(start_ID, end_ID, group_size, 
                                        label_list, label_list_level, both_label_indices, return_coarse=False):
    try:
        assert label_list_level==0 or label_list_level==1
    except AssertionError:
        raise ValueError("balanced_label should be either 0 for fine label or 1 for coarse label in both_label_indices.")

    
    prop_label_dict = both_label_indices[label_list_level]
        
    per_label_group_size = math.ceil(group_size/len(label_list))
    extra_samples_num = len(label_list) * per_label_group_size - group_size
    
    print("Number of samples per label = ", per_label_group_size, 
          "number of extra samples because of division round up = ", extra_samples_num)
    
    
    per_label_IDs = dict()
    per_label_len = dict()
    for lbl in label_list:
        per_label_IDs[lbl] = [-1]*per_label_group_size
        per_label_len[lbl] = 0
        
    
    retrieved_IDs = [-1] * group_size
    retrieved_labels = [-1] * group_size
    
    # we don't randomly sample to preserve the homogeneity of the samples to be able to see distribution change
    total_retrieved = 0
    for i, x in enumerate(config.get_all_labelled_IDs()[start_ID:end_ID]):
        
        label_num = x[label_list_level]
        
        if label_num not in label_list:
            continue
        
        x_id = x[2]
        x_label = x[int(return_coarse)]
        label_num_len = per_label_len[label_num]
        
        if label_num_len < per_label_group_size:
            per_label_IDs[label_num][per_label_len[label_num]] = x_id
            retrieved_IDs[total_retrieved] = x_id
            retrieved_labels[total_retrieved] = x_label
            per_label_len[label_num] += 1
            total_retrieved +=1 
            if total_retrieved %100==0:
                print("retrieved ", total_retrieved, "balanced samples")
        
        if total_retrieved == group_size:
            break
    
    return retrieved_IDs, retrieved_labels, per_label_IDs, per_label_len
    

def find_filelist_for_ids(idlist, return_dict = False):
    prefix = config.file_prefix
    step = config.file_step_size
    
    idlist_pointer = 0
    filelist_pointer = 0
    
    output_filelist = []
    
    stop = False
    
    idlist = sorted(idlist)
    
    for itemid in idlist:
        file_startId = itemid - (itemid%step)
        output_filelist.append([file_startId,
                                "{}{}_{}.pkl".format(prefix, file_startId,(file_startId+step)),
                                (itemid%step)])
        
    idfilelist = list(zip(idlist, output_filelist))
    
    # output format list of (ID, [first_id_in_file, filepath, offset_to_ID])
    if not return_dict: return idfilelist
    
    file2idlist = dict()

    for xid, [startid, path, offset] in idfilelist:
        
        if path not in file2idlist:
            file2idlist[path] = []
        
        file2idlist[path].append([xid, offset])
    
    # output format dictionary of file path -> list of [id, offset-in-file]
    return file2idlist
    

def load_data_for_ids(idlist, debug=False):
    file2idlist = find_filelist_for_ids(idlist, True)
    
    X_ppis = [None] * len(idlist)
    
    index = 0
    filenumber = 0
    flag = False
    for key in file2idlist:
        filenumber+=1
        offsetlist = file2idlist[key]
        
        with open(key, 'rb') as file:
            f_ppis = pkl.load(file)
            
            # filter f_ppis by offset
            for xid,offset in offsetlist:
                try:
                    X_ppis[index] = f_ppis[offset]
                    # if lists are iterated by formation order, the order of the ids should be preserved
                    if debug: assert idlist.index(xid) == index
                    index+=1
                    
                except AssertionError:
                    if not flag: 
                        print("WARNING: returned data items are not in the order of input idlist")
                        flag = True
                    
            
            
    return X_ppis


def calculate_cesnet_mean_for_ids(idlist, IAT):
    total_count = 0
    total_sum = 0
    max_axis = -1
    min_axis = 30000000

    WEIRD_FILES = []
    
    axisname = ""
    
    if IAT:
        axis = 0 
        axisname = "IAT"
    else:
        axis = 2 
        axisname = "pktsize"
        
    file2ids = find_filelist_for_ids(idlist, True)
    
    # we are not using load_data_for_ids because we want to go file by file to not exhaust the memory

    for path in file2ids:
        
        offsetlist = [offset for xid,offset in file2ids[path]]
        
        with open(path, 'rb') as f:

            X_file = pkl.load(f)
            
            ps_file_for_ids = [x[axis] for i,x in enumerate(X_file) if i in offsetlist]

            ps_max_file = max([np.max(ps) for ps in ps_file_for_ids])
            ps_min_file = min([np.min(ps) for ps in ps_file_for_ids])
            
            ps_sum_file = sum([np.sum(ps) for ps in ps_file_for_ids])

            if ps_min_file < 0:
                WEIRD_FILES.append(path)

            if max_axis < ps_max_file:
                max_axis = ps_max_file

            if min_axis > ps_min_file:
                min_axis = ps_min_file

            count_file = sum([len(x) for x in ps_file_for_ids])
            
            total_count += count_file
            total_sum +=ps_sum_file

            print(" Done with ", path[len(config.file_prefix):],end=";")
            
    print("max and min {}: ".format(axisname), max_axis, min_axis)
    print("total packets in dataset: ", total_count)
    print("{} files with min {} smaller than 0 :".format(len(WEIRD_FILES), axisname), WEIRD_FILES)
    
    mean_axis = total_sum/total_count

    return  min_axis, max_axis, mean_axis


# calculate standard dev
def calculate_cesnet_std_for_ids(idlist, mean, IAT):
    total_count = 0
    total_sum = 0
    
    axisname = ""
    if IAT:
        axis = 0 
        axisname = "IAT"
    else:
        axis = 2 
        axisname = "pktsize"
                      
    file2ids = find_filelist_for_ids(idlist, True)
    
    # we are not using load_data_for_ids because we want to go file by file to not exhaust the memory

    for path in file2ids:
        
        offsetlist = [offset for xid,offset in file2ids[path]]
        
        with open(path, 'rb') as f:

            X_file = pkl.load(f)
            
            ps_file_for_ids = [x[axis] for i,x in enumerate(X_file) if i in offsetlist]

            ps_max_file = max([np.max(ps) for ps in ps_file_for_ids])
            ps_min_file = min([np.min(ps) for ps in ps_file_for_ids])
            
            ps_sum_file = sum([np.sum(np.power( x - mean , 2)) for x in ps_file_for_ids])

            count_file = sum([len(x) for x in ps_file_for_ids])
            
            total_count += count_file
            total_sum +=ps_sum_file

            print(" Done with ", path[len(config.file_prefix):],end=";")

    standard_deviation = np.sqrt(total_sum / total_count)

    return standard_deviation
