
from math import log
import operator
import pickle

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def calc_shannon_ent(data_set):

    num = len(data_set)

    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num
        shannon_ent = shannon_ent - prob * log(prob, 2)
    return shannon_ent

def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set

def make_leaf(subset):
	outcomes = [row[-1] for row in subset]
	return max(set(outcomes), key=outcomes.count)
# ID3
def choose_best_feature_to_split(data_set):
    num_feature = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0
    best_feature_idx = -1
    for feature_idx in range(num_feature):
        feature_val_list = [number[feature_idx] for number in data_set]
        unique_feature_val_list = set(feature_val_list) 
        new_entropy = 0
        for feature_val in unique_feature_val_list:
            sub_data_set = split_data_set(data_set, feature_idx, feature_val)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set) #
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_idx = feature_idx
    return best_feature_idx

# C4.5
def choose_best_feature_to_split_ratio(data_set):

    num_feature = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set) 
    best_info_gain_ratio = 0.0
    best_feature_idx = -1
    for feature_idx in range(num_feature):
        feature_val_list = [number[feature_idx] for number in data_set]  
        unique_feature_val_list = set(feature_val_list)
        new_entropy = 0
        split_info = 0.0
        for value in unique_feature_val_list:
            sub_data_set = split_data_set(data_set, feature_idx, value)
            prob = len(sub_data_set) / float(len(data_set))  
            new_entropy += prob * calc_shannon_ent(sub_data_set) 
            split_info += -prob * log(prob, 2)
        info_gain = base_entropy - new_entropy  
        if split_info == 0: 
            continue
        info_gain_ratio = info_gain / split_info

        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_feature_idx = feature_idx
    print("best Feature to split is ", best_feature_idx)
    return best_feature_idx

def create_ID3tree(data_set, labels, max_depth, depth):
        
    class_list = [sample[-1] for sample in data_set] 

    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[0]
    if depth >= max_depth:
        return make_leaf(data_set)
    if len(data_set[0]) == 1:
        return majority_cnt((class_list))

    best_feature_idx = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feature_idx]
    print("creating ID3 Tree", labels[best_feature_idx])
    my_tree = {best_feat_label: {}}
    del (labels[best_feature_idx]) 
    feature_values = [example[best_feature_idx] for example in data_set]
    unique_feature_values = set(feature_values)
    if depth<max_depth:
        for feature_value in unique_feature_values:
            sub_labels = labels[:]

            sub_data_set = split_data_set(data_set, best_feature_idx, feature_value)
            my_tree[best_feat_label][feature_value] = create_ID3tree(sub_data_set, sub_labels, max_depth, depth+1)
    return my_tree

def create_C45tree(data_set, labels, max_depth, depth):
    class_list = [sample[-1] for sample in data_set] 

    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[-1]
    if depth >= max_depth:
        return majority_cnt((class_list))
    if len(data_set[0]) == 1:
        return majority_cnt((class_list))

    best_feature_idx = choose_best_feature_to_split_ratio(data_set)
    best_feat_label = labels[best_feature_idx]
    print("creating C45 Tree", labels[best_feature_idx])
    my_tree = {best_feat_label: {}}
    del (labels[best_feature_idx]) 
    feature_values = [example[best_feature_idx] for example in data_set]
    unique_feature_values = set(feature_values)
    if depth < max_depth:
        for feature_value in unique_feature_values:
            sub_labels = labels[:]

            sub_data_set = split_data_set(data_set, best_feature_idx, feature_value)
            my_tree[best_feat_label][feature_value] = create_C45tree(sub_data_set, sub_labels, max_depth, depth+1)
    return my_tree

