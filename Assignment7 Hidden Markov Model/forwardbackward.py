# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：forwardbackward.py.py
@Author  ：Tianye Song
@Date    ：11/11/21 17:52:27 
"""
import csv
import sys
import numpy as np
import argparse

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser(description="Hidden Markov Model Forward Backward Algorithm")
parser.add_argument('num', type=int)
parser.add_argument('validation_input', type=str)
parser.add_argument('index_to_word', type=str)
parser.add_argument('index_to_tag', type=str)
parser.add_argument('hmminit', type=str)
parser.add_argument('hmmemit', type=str)
parser.add_argument('hmmtrans', type=str)
parser.add_argument('predicted_file', type=str)
parser.add_argument('metric_file', type=str)


def read_file(input_path, deli='\t'):
    """
    read csv file
    :param input_path:
    :param deli:
    :return:
    """
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli)
        data_array = []
        para = []
        for row in raw:
            if len(row) != 0:
                para.append(row)
            else:
                data_array.append(para)
                para = []
        data_array.append(para)
        return data_array


def read_matrix(input_path, deli=' '):
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli, quoting=csv.QUOTE_NONNUMERIC)
        result = np.array(list(raw))
        return result


def read_map(input_path):
    """
    read map file
    :param input_path:
    :return:
    """
    with open(input_path, 'r', encoding='UTF-8') as file:
        map = dict()
        for i, row in enumerate(file.readlines()):
            map[row.rstrip()] = i
        return map


def write_file(output, output_path, method='w', flag="text"):
    """
    :param output:
    :param output_path:
    :param method: create a new file or append
    :param flag: write number or text
    :return:
    """
    if flag == "text":
        with open(output_path, method) as file:
            for sen in output:
                for word in sen:
                    file.write(word[0])
                    file.write("\t")
                    file.write(word[1])
                    file.write('\n')
                file.write("\n")
    else:
        # flag == number
        with open(output_path, method) as file:
            file.write(output)
            file.write('\n')


class HMM:
    def __init__(self, hmminit, hmmtrans, hmmemit):
        self.hmminit = np.log(hmminit)
        self.hmmtrans = np.log(hmmtrans)
        self.hmmemit = np.log(hmmemit)

    def forward(self, t, x, forward_matrix):
        if t == 0:
            return self.hmminit + self.hmmemit[:, x]
            # after taking log
        else:
            a = self.hmmemit[:, x].reshape(1, -1)
            b = self.hmmtrans
            c = forward_matrix[:, t - 1].reshape(-1, 1)
            d = b+c
            e = np.max(d, axis=0).reshape(1, -1)
            f = d-e
            g = np.log(np.sum(np.exp(f), axis=0)).reshape(1, -1)
            return a + g + e

    def backward(self, t, sentence, backward_matrix):
        if t == len(sentence) - 1:
            return np.zeros([1, self.hmminit.shape[1]])
        else:
            a = (self.hmmemit[:, sentence[t + 1]] + backward_matrix[:, t + 1]).reshape(1, -1)
            b = (self.hmmtrans + a)
            c = np.max(b, axis=1).reshape(-1, 1)
            return np.log(np.sum(np.exp(b - c), axis=1)).reshape(-1, 1) + c

    def predict(self, obs):
        result_tag_list = []
        log = 0
        for sen in obs:
            forward_matrix = np.zeros([self.hmminit.shape[1], len(sen)])
            backward_matrix = np.zeros([self.hmminit.shape[1], len(sen)])
            for i in range(0, len(sen)):
                forward_matrix[:, i] = self.forward(i, sen[i], forward_matrix).reshape(1, -1)
                backward_matrix[:, len(sen) - 1 - i] = self.backward(len(sen) - 1 - i, sen, backward_matrix).reshape(1, -1)
            x = np.argmax(forward_matrix + backward_matrix, axis=0).tolist()
            result_tag_list.append(list(zip(sen, x)))
            a = forward_matrix[:, -1]
            b = np.max(forward_matrix[:, -1])
            c = a-b
            log += np.log(np.sum(np.exp(c))) + b
        log = log / len(result_tag_list)
        return result_tag_list, log


if __name__ == '__main__':
    args = parser.parse_args()
    validation_data = read_file(args.validation_input)[:args.num]

    word_to_index = read_map(args.index_to_word)
    tag_to_index = read_map(args.index_to_tag)
    index_to_tag = dict(zip(tag_to_index.values(), tag_to_index.keys()))
    index_to_word = dict(zip(word_to_index.values(), word_to_index.keys()))

    validation_data_index = [[word_to_index.get(y[0]) for y in x] for x in validation_data]
    hmminit = read_matrix(args.hmminit).reshape(1, -1)
    hmmtrans = read_matrix(args.hmmtrans)
    hmmemit = read_matrix(args.hmmemit)

    hmm = HMM(hmminit, hmmtrans, hmmemit)
    validation_data_predict_index, log_likelihood = hmm.predict(validation_data_index)
    validation_data_predict = [[[index_to_word.get(y[0]), index_to_tag.get(y[1])] for y in x] for x in
                               validation_data_predict_index]
    acc = 0
    total = 0
    for (para0, para1) in zip(validation_data,validation_data_predict):
        for word0, word1 in zip(para0, para1):
            total += 1
            if word0[1] == word1[1]:
                acc += 1
    acc_rate = acc/total
    write_file(validation_data_predict, args.predicted_file)
    str1 = "Average Log-Likelihood: {}".format(log_likelihood)
    str2 = "Accuracy: {}".format(acc_rate)
    write_file(str1, args.metric_file, flag='number')
    write_file(str2, args.metric_file, 'a', flag='number')