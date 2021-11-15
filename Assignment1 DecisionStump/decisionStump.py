# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：decisionStump.py
@Author  ：Tianye Song
@Date    ：9/5/21 12:28:48 
"""

import collections
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description='Decision Tree Stump')
parser.add_argument('train_input', type=str)
parser.add_argument('test_input', type=str)
parser.add_argument('split_index', type=int)
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
    data = np.array(data)
    return feature, data


def write_file(output, output_path, method='x'):
    with open(output_path, method) as file:
        for line in output:
            file.write(line)
            file.write('\n')


def write_number(output, output_path, method='x'):
    with open(output_path, method) as file:
        file.write(output)
        file.write('\n')


class DecisonStump:

    def __init__(self, depth, split, metrics_out_path):
        """
        initiate class
        :param depth:
        :param split:
        """
        self.depth = depth
        self.split = split
        self.metrics = metrics_out_path

    def split_data(self, data, feature_index):
        label_set = np.unique(data[:, feature_index])
        index0 = np.where(data[:, feature_index] == label_set[0])
        index1 = np.where(data[:, feature_index] == label_set[1])
        data0 = data[index0]
        data1 = data[index1]
        return label_set[0], label_set[1], data0, data1

    def major_vote(self, data):
        vote = collections.Counter(data[:, -1]).most_common(1)[0][0]
        return vote

    def cal_accur(self, ori, pred):
        accuracy = np.sum(ori != pred) / np.size(ori, 0)
        return accuracy

    def train(self, train_input, train_output):
        feature_name, train = read_file(train_input)
        tag0, tag1, data0, data1 = self.split_data(train, self.split)

        data0_label = self.major_vote(data0)
        data1_label = self.major_vote(data1)

        self.stump = {tag0: data0_label, tag1: data1_label}

        ori_label = train[:, -1]
        predict_label = [self.stump[i] for i in train[:, self.split]]

        write_file(predict_label, train_output)
        accuracy = self.cal_accur(ori_label, predict_label)
        sen = 'error(train): {}'.format(accuracy)
        write_number(sen, self.metrics, 'x')

    def test(self, test_input, test_output):
        feature_name, test = read_file(test_input)
        predict_label = [self.stump[i] for i in test[:, self.split]]
        ori_label = test[:, -1]
        write_file(predict_label, test_output)
        accuracy = self.cal_accur(ori_label, predict_label)
        sen = 'error(test): {}'.format(accuracy)
        write_number(sen, self.metrics, 'a')


if __name__ == '__main__':
    args = parser.parse_args()
    stump = DecisonStump(1, args.split_index, args.metrics_out)
    stump.train(args.train_input, args.train_output)
    stump.test(args.test_input, args.test_output)
