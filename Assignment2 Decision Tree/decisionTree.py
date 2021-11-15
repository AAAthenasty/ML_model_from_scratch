# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：decisionTree_ori.py
@Author  ：Tianye Song
@Date    ：9/14/21 14:11:55 
"""
import csv
import collections
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Decision Tree')
parser.add_argument('train_input', type=str)
parser.add_argument('test_input', type=str)
parser.add_argument('max_depth', type=int)
parser.add_argument('train_output', type=str)
parser.add_argument('test_output', type=str)
parser.add_argument('metrics_out', type=str)


def read_file(input_path, deli='\t'):
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli)
        data = []
        feature = []
        i = 0
        for row in raw:
            if i == 0:
                feature = row
                i = i + 1
            else:
                data.append(row)
    data_array = np.array(data)
    return feature, data_array


def write_file(output, output_path, method='x'):
    with open(output_path, method) as file:
        for line in output:
            file.write(line)
            file.write('\n')


def write_number(output, output_path, method='x'):
    with open(output_path, method) as file:
        file.write(output)
        file.write('\n')


def major_vote(sample_data):
    result = collections.Counter(sample_data[:, -1]).most_common(2)
    if len(result) == 1:
        return result[0][0]
    else:
        if result[0][1] == result[1][1]:
            result.sort()
            return result[1][0]
        else:
            return result[0][0]


def cal_entropy(sample_data):
    label_set = np.unique(sample_data[:, -1])
    len0 = sample_data[np.where(sample_data[:, -1] == label_set[0])].shape[0]
    if len(label_set) == 1:
        return 0
    else:
        len1 = sample_data[np.where(sample_data[:, -1] == label_set[1])].shape[0]
        p1 = len0 / (len0 + len1)
        p2 = len1 / (len0 + len1)
        entropy = -p1 * np.log2(p1) - p2 * np.log2(p2)
        return entropy


def cal_error(ori, pred):
    error = np.sum(ori != pred) / np.size(ori, 0)
    return error


def find_max_info_gain(data):
    ori_entropy = cal_entropy(data)
    min_entropy = 1
    min_entropy_index = None
    for i in range(data.shape[1] - 1):
        label_set = np.unique(data[:, i])
        data0 = data[np.where(data[:, i] == label_set[0])]
        p0 = (data0.shape[0]) / (data.shape[0])
        entropy0 = cal_entropy(data0)
        if len(label_set) == 1:
            entropy = entropy0
        else:
            data1 = data[np.where(data[:, i] == label_set[1])]
            p1 = (data1.shape[0]) / (data.shape[0])
            entropy1 = cal_entropy(data1)
            entropy = p0 * entropy0 + p1 * entropy1
        if entropy < min_entropy:
            min_entropy = entropy
            min_entropy_index = i
    return ori_entropy - min_entropy, min_entropy_index


class TreeNode:

    def __init__(self, max_depth, depth=0, node_feature=None, node_feature_value=None, label_set=None):
        self.left_node = None
        self.right_node = None
        self.left_node_feature = None
        self.right_node_feature = None
        self.depth = depth
        self.feature_index = None
        self.max_depth = max_depth
        self.label = None
        self.node_feature = node_feature
        self.node_feature_value = node_feature_value
        self.label_set = label_set

    def split_data(self, data, feature_index):
        label_set = np.unique(data[:, feature_index])
        index0 = np.where(data[:, feature_index] == label_set[0])
        index1 = np.where(data[:, feature_index] == label_set[1])
        data0 = data[index0]
        data1 = data[index1]
        return label_set[0], label_set[1], data0, data1

    def train(self, data, feature_list):
        if self.depth == 0:
            self.label_set = np.unique(data[:, -1])
        self.print_tree_layout(data)
        if len(np.unique(data[:, -1])) == 1:
            self.label = np.unique(data[:, -1])[0]
        elif self.depth >= self.max_depth:
            self.label = major_vote(data)
        else:
            info_gain, feature_index = find_max_info_gain(data)
            if info_gain > 0:
                node_feature = feature_list[feature_index]
                self.feature_index = feature_index
                self.left_node_feature, self.right_node_feature, \
                    left_data, right_data = self.split_data(data, self.feature_index)
                self.left_node = TreeNode(self.max_depth, self.depth + 1, node_feature, self.left_node_feature,
                                          self.label_set)
                self.right_node = TreeNode(self.max_depth, self.depth + 1, node_feature, self.right_node_feature,
                                           self.label_set)
                self.left_node.train(left_data, feature_list)
                self.right_node.train(right_data, feature_list)
            else:
                self.label = major_vote(data)

    def predict_point(self, point):
        if self.label:
            return np.array(self.label, dtype='<U100')
        elif point[self.feature_index] == self.left_node_feature:
            return self.left_node.predict_point(point)
        else:
            return self.right_node.predict_point(point)

    def predict(self, data):
        return np.apply_along_axis(self.predict_point, 1, data)

    def print_tree_layout(self, data):
        pair_list = []
        for label in self.label_set:
            pair_list.append("{} {}".format(len(data[np.where(data[:, -1] == label)]), label))
        result = "/".join(pair_list)
        if self.depth == 0:
            print("[" + result + "]")
        else:
            print(self.depth * "| " + self.node_feature + ": " + self.node_feature_value + "[" + result + "]")


if __name__ == '__main__':
    args = parser.parse_args()
    train_feature, train_data = read_file(args.train_input)
    test_feature, test_data = read_file(args.test_input)
    stump = TreeNode(args.max_depth)
    stump.train(train_data, train_feature)
    train_predict = stump.predict(train_data)
    test_predict = stump.predict(test_data)
    write_file(train_predict, args.train_output)
    write_file(test_predict, args.test_output)
    train_error = cal_error(train_data[:, -1], train_predict)
    test_error = cal_error(test_data[:, -1], test_predict)
    sen0 = "error(train): {}".format(train_error)
    sen1 = "error(test): {}".format(test_error)
    write_number(sen0, args.metrics_out)
    write_number(sen1, args.metrics_out, 'a')
