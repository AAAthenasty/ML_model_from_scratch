# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：plot_lr.py
@Author  ：Tianye Song
@Date    ：10/9/21 23:30:28
"""

import csv

import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='NLP feature engineering')
parser.add_argument('formatted_train_input', type=str)
parser.add_argument('formatted_validation_input', type=str)
parser.add_argument('formatted_test_input', type=str)
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

    def train(self, train_feature, train_tag, validation_feature, validation_tag):
        train_feature = np.insert(train_feature, 0, 1, axis=1)
        validation_feature = np.insert(validation_feature, 0, 1, axis=1)
        train_size = train_feature.shape[0]
        valid_size = validation_feature.shape[0]
        self.weight = np.zeros(train_feature.shape[1])
        metric_list = []
        for num in range(0, self.epoch_num):
            self.train_epoch(train_feature, train_tag, train_size)
            obj = self.cal_obj(validation_feature, validation_tag, valid_size)
            metric_list.append([num + 1, obj])
        return np.array(metric_list)

    def train_epoch(self, data_feature, data_tag, size):
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

    def cal_obj(self, data_feature, data_tag, size):
        obj = (-np.dot(data_tag, np.dot(data_feature, self.weight.T).T) +
               np.sum(np.log(1 + np.exp(np.dot(data_feature, self.weight.T))))) / size
        return obj


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
    valid_data = read_file(args.formatted_validation_input)
    test_data = read_file(args.formatted_test_input)
    for step_size in [0.001,0.01,0.1]:
        lr = LogisticRegression(args.num_epoch, step=step_size)
        result = lr.train(train_data[:, 1:], train_data[:, 0], valid_data[:, 1:], valid_data[:, 0])
        plt.plot(result[:, 0], result[:, 1], label="step_size={}".format(step_size))
    plt.title("Bag of words")
    plt.xlabel("Epoch num")
    plt.ylabel("Negative log likelihood")
    plt.legend()
    plt.show()



    # lr.train(train_data[:, 1:], train_data[:, 0], valid_data[:, 1:], valid_data[:, 0])
    # print("train")
    # print(lr.cal_error(train_data[:, 0], lr.predict(train_data[:, 1:])))
    # print("test")
    # print(lr.cal_error(test_data[:, 0], lr.predict(test_data[:, 1:])))
