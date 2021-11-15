# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：lr.py
@Author  ：Tianye Song
@Date    ：10/9/21 11:16:34 
"""
import csv

import ast
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='NLP feature engineering')
parser.add_argument('formatted_train_input', type=str)
parser.add_argument('formatted_validation_input', type=str)
parser.add_argument('formatted_test_input', type=str)
parser.add_argument('dict_input', type=str)

parser.add_argument('train_out', type=str)
parser.add_argument('test_out', type=str)
parser.add_argument('metrics_out', type=str)
parser.add_argument('num_epoch', type=int)


class LogisticRegression:
    def __init__(self, epoch_num, step):
        self.epoch_num = epoch_num
        self.weight = np.array([1, 1])
        self.step = step

    def sgd(self, input_feature, input_tag, data_size):
        # input is one sample
        gradient = -np.multiply(input_feature, input_tag - np.exp(np.dot(self.weight.T, input_feature)) / (
                1 + np.exp(np.dot(self.weight.T, input_feature))))
        self.weight = self.weight - self.step * gradient / data_size

    def train(self, data_feature, data_tag):
        data_feature = np.insert(data_feature, 0, 1, axis=1)
        size = data_feature.shape[0]
        self.weight = np.zeros(data_feature.shape[1])
        for num in range(0, self.epoch_num):
            for i, single_feature in enumerate(data_feature):
                self.sgd(single_feature, data_tag[i], size)

    def predict(self, input_data):
        input_data = np.insert(input_data, 0, 1, axis=1)
        result = 1 / (1 + np.exp(-np.dot(input_data, self.weight.T)))

        result = np.array([1.0 if x > 0.5 else 0.0 for x in result])
        return result

    def cal_error(self, ori, pred):
        err_rate = np.round(np.sum(ori != pred) / np.size(ori, 0), 6)
        return err_rate


def read_file(input_path, deli='\t'):
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli, quoting=csv.QUOTE_NONNUMERIC)
        data_array = np.array(list(raw))
    return data_array


def write_file(output, output_path, method='x', flag="text"):
    if flag == "text":
        with open(output_path, method) as file:
            for line in output:
                file.write(line)
                file.write('\n')
    else:
        # flag == number
        with open(output_path, method) as file:
            file.write(output)
            file.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    train_data = read_file(args.formatted_train_input)
    test_data = read_file(args.formatted_test_input)
    lr = LogisticRegression(args.num_epoch, step=0.01)
    lr.train(train_data[:, 1:], train_data[:, 0])

    train_predict = lr.predict(train_data[:, 1:])
    test_predict = lr.predict(test_data[:, 1:])
    write_file( [str(int(x)) for x in train_predict], args.train_out)
    write_file([str(int(x)) for x in test_predict], args.test_out)

    train_error = lr.cal_error(train_data[:, 0], train_predict)
    test_error = lr.cal_error(test_data[:, 0], test_predict)
    sen0 = "error(train): {}".format(train_error)
    sen1 = "error(test): {}".format(test_error)
    write_file(sen0, args.metrics_out, flag="number")
    write_file(sen1, args.metrics_out, 'a', flag="number")