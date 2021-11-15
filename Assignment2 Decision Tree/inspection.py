# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：inspection.py
@Author  ：Tianye Song
@Date    ：9/14/21 14:11:25 
"""

import argparse
import numpy as np
import collections
import csv


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
    vote = collections.Counter(sample_data[:, -1]).most_common(1)[0][0]
    return vote


def cal_entropy(sample_data):
    label_set = np.unique(sample_data[:, -1])
    len0 = sample_data[np.where(sample_data[:, -1] == label_set[0])].shape[0]
    if len(label_set) == 1:
        return 0
    else:
        len1 = sample_data[np.where(sample_data[:, -1] == label_set[1])].shape[0]
        p1 = len0 / (len0 + len1)
        p2 = len1 / (len0 + len1)
        entro = -p1 * np.log2(p1) - p2 * np.log2(p2)
        return entro


def cal_error(ori, pred):
    error_rate = np.sum(ori != pred) / np.size(ori, 0)
    return error_rate


def find_max_info_gain(data):
    ori_entropy = cal_entropy(data)
    min_entro = 1
    min_entro_index = None
    for i in range(data.shape[1]):
        label_set = np.unique(data[:, i])
        data0 = data[np.where(data[:, i] == label_set[0])]
        data1 = data[np.where(data[:, i] == label_set[1])]
        p0 = (data0.shape[0])/(data0.shape[0]+data1.shape[0])
        p1 = (data1.shape[0])/(data0.shape[0]+data1.shape[0])
        entro0 = cal_entropy(data0)
        entro1 = cal_entropy(data1)
        entro = p0*entro0+p1*entro1
        if entro < min_entro:
            min_entro = entro
            min_entro_index = i
    return ori_entropy-min_entro, min_entro_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decision Tree')
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    feature_name, data = read_file(args.input_path)
    entropy = cal_entropy(data)
    label = major_vote(data)
    error = cal_error(data[:, -1], label)
    sen0 = 'entropy: {}'.format(entropy)
    sen1 = 'error: {}'.format(error)
    write_number(sen0, args.output_path)
    write_number(sen1, args.output_path, 'a')
