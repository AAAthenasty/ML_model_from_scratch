# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：neuralnet.py
@Author  ：Tianye Song
@Date    ：10/18/21 23:51:53 
"""
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Neural Network')
parser.add_argument('train_input', type=str)
parser.add_argument('validation_input', type=str)

parser.add_argument('train_out', type=str)
parser.add_argument('validation_out', type=str)
parser.add_argument('metrics_out', type=str)

parser.add_argument('num_epoch', type=int)
parser.add_argument('hidden_units', type=int)
parser.add_argument('init_flag', type=int, help="1 for randomly 2 for zeros")
parser.add_argument('learning_rate', type=float)


def read_file(input_path, deli='\t'):
    """
    read csv file
    :param input_path:
    :param deli:
    :return:
    """
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli, quoting=csv.QUOTE_NONNUMERIC)
        data_array = np.array(list(raw))
    return data_array


def write_file(output, output_path, method='x', flag="text"):
    """
    :param output:
    :param output_path:
    :param method: create a new file or append
    :param flag: write number or text
    :return:
    """
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


class NeuralNetwork:
    def __init__(self, num_epoch, hidden_units, learning_rate, init_flag, train_output, valid_output, metric_file):
        self.lr = learning_rate
        self.hidden_units = hidden_units
        self.init_flag = init_flag
        self.num_epoch = num_epoch
        self.epsilon = 0.00001
        self.first_layer_intermediate = None
        self.second_layer_intermediate = None
        self.first_layer_para = None
        self.second_layer_para = None
        self.tag_shape = None
        self.feature_shape = None
        self.train_output = train_output
        self.valid_output = valid_output
        self.metric_file = metric_file

    def train(self, train_data_feature, train_data_tag, valid_data_feature, valid_data_tag):
        train_data_feature = np.insert(train_data_feature, 0, 1, axis=1)
        train_data_tag = train_data_tag.astype(int)
        train_data_tag_onehot = np.zeros((train_data_tag.size, np.max(train_data_tag) + 1))
        train_data_tag_onehot[np.arange(train_data_tag.size), train_data_tag] = 1

        self.feature_shape = train_data_feature.shape[1]
        self.tag_shape = train_data_tag_onehot.shape[1]

        valid_data_feature = np.insert(valid_data_feature, 0, 1, axis=1)
        valid_data_tag = valid_data_tag.astype(int)
        valid_data_tag_onehot = np.zeros((valid_data_tag.size, self.tag_shape))
        valid_data_tag_onehot[np.arange(valid_data_tag.size), valid_data_tag] = 1

        self.para_init()

        for i in np.arange(0, self.num_epoch):
            self.sgd(train_data_feature, train_data_tag_onehot)

            train_loss = self.loss_func(self.nnforward(train_data_feature)[1], train_data_tag_onehot)
            valid_loss = self.loss_func(self.nnforward(valid_data_feature)[1], valid_data_tag_onehot)

            train_metric = "epoch={} crossentropy(train) : {}".format(i + 1, train_loss)
            valid_metric = "epoch={} crossentropy(validation) : {}".format(i + 1, valid_loss)

            if i == 0:
                write_file(train_metric, self.metric_file, 'x', "number")
            else:
                write_file(train_metric, self.metric_file, 'a', "number")
            write_file(valid_metric, self.metric_file, 'a', "number")

        train_data_predict = np.argmax(self.nnforward(train_data_feature)[1], axis=1)
        valid_data_predict = np.argmax(self.nnforward(valid_data_feature)[1], axis=1)
        self.write_result(train_data_predict, train_data_tag, valid_data_predict, valid_data_tag)

    def write_result(self, train_data_predict, train_data_tag, valid_data_predict, valid_data_tag):
        train_error = self.cal_error(train_data_tag, train_data_predict)
        valid_error = self.cal_error(valid_data_tag, valid_data_predict)

        train_error_sen = "error(train): {}".format(train_error)
        valid_error_sen = "error(validation): {}".format(valid_error)
        write_file(train_error_sen, self.metric_file, 'a', "number")
        write_file(valid_error_sen, self.metric_file, 'a', "number")
        write_file([str(x) for x in train_data_predict], self.train_output)
        write_file([str(x) for x in valid_data_predict], self.valid_output)

    def para_init(self):
        """
        parameter initialization
        :return:
        """
        if self.init_flag == 1:
            self.first_layer_para = np.random.random([self.hidden_units, self.feature_shape]) / 5 - 0.1
            self.second_layer_para = np.random.random([self.tag_shape, self.hidden_units + 1]) / 5 - 0.1
        else:
            self.first_layer_para = np.zeros([self.hidden_units, self.feature_shape])
            self.second_layer_para = np.zeros([self.tag_shape, self.hidden_units + 1])
        self.first_layer_para[:0] = 0
        self.second_layer_para[:0] = 0
        self.first_layer_intermediate = np.zeros([self.hidden_units, self.feature_shape])
        self.second_layer_intermediate = np.zeros([self.tag_shape, self.hidden_units + 1])

    def nnforward(self, feature):
        a = np.dot(self.first_layer_para, feature.T).T
        z = np.insert(1 / (1 + np.exp(-a)), 0, 1, axis=1)
        b = np.dot(self.second_layer_para, z.T).T
        y_hat = np.exp(b) / np.reshape(np.sum(np.exp(b), axis=1), (-1, 1))
        return z, y_hat

    # 这个地方补充
    def nnbackward(self, feature, tag):
        """
        backward parameter update
        :param feature:
        :param tag:
        :return:
        """
        z, y_hat = self.nnforward(feature)

        b_g = np.reshape(y_hat - tag, (1, -1))
        second_layer_gradient = np.dot(b_g.T, np.reshape(z, (1, -1)))
        z_g = np.dot(b_g, self.second_layer_para[:, 1:])
        z_a = z[:, 1:] * (1 - z[:, 1:])
        a_g = z_g * z_a
        first_layer_gradient = np.dot(a_g.T, feature)
        return first_layer_gradient, second_layer_gradient

    def sgd(self, train_data_feature, train_data_tag):
        """
        stochastic gradient descent
        :param train_data_feature:
        :param train_data_tag:
        :return:
        """
        for (x, y) in zip(train_data_feature, train_data_tag):
            first_layer_gradient, second_layer_gradient = self.nnbackward(np.reshape(x, (1, -1)), y)
            self.adagrad(first_layer_gradient, second_layer_gradient)

    def adagrad(self, first_layer_gradient, second_layer_gradient):
        """
        :param first_layer_gradient:
        :param second_layer_gradient:
        :return:
        """
        self.first_layer_intermediate = self.first_layer_intermediate + first_layer_gradient * first_layer_gradient
        self.second_layer_intermediate = self.second_layer_intermediate + second_layer_gradient * second_layer_gradient
        self.first_layer_para = self.first_layer_para - (
                self.lr / np.sqrt(self.first_layer_intermediate + self.epsilon) * first_layer_gradient)
        self.second_layer_para = self.second_layer_para - (
                self.lr / np.sqrt(self.second_layer_intermediate + self.epsilon) * second_layer_gradient)

    def predict(self, feature):
        z, y_hat = self.nnforward(feature)
        y_predict = np.zeros(self.tag_shape)
        y_predict[np.argmax(y_hat)] = 1
        return y_predict

    def loss_func(self, y_predict, y_true):
        cross_entropy = -np.sum(y_true * np.log(y_predict)) / y_true.shape[0]
        return cross_entropy

    def cal_error(self, ori, pred):
        err_rate = np.round(np.sum(ori != pred) / np.size(ori, 0), 6)
        return err_rate


if __name__ == '__main__':
    args = parser.parse_args()
    train_data = read_file(args.train_input, deli=",")
    validation_data = read_file(args.validation_input, deli=",")
    NN = NeuralNetwork(args.num_epoch, args.hidden_units, args.learning_rate, args.init_flag, args.train_out,
                       args.validation_out, args.metrics_out)
    NN.train(train_data[:, 1:], train_data[:, 0], validation_data[:, 1:], validation_data[:, 0])
