# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：learnhmm.py
@Author  ：Tianye Song
@Date    ：11/11/21 14:34:52 
"""
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Hidden Markov Model Parameter")
parser.add_argument('num', type=int)
parser.add_argument('train_input', type=str)
parser.add_argument('index_to_word', type=str)
parser.add_argument('index_to_tag', type=str)
parser.add_argument('hmminit', type=str)
parser.add_argument('hmmemit', type=str)
parser.add_argument('hmmtrans', type=str)


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
    train_data = read_file(args.train_input)[:args.num]
    index_to_word = read_map(args.index_to_word)
    index_to_tag = read_map(args.index_to_tag)
    train_data_index = [[[index_to_word.get(y[0]), index_to_tag.get(y[1])] for y in x] for x in train_data]
    init = np.ones(len(index_to_tag))
    trans = np.ones([len(index_to_tag), len(index_to_tag)])
    emit = np.ones([len(index_to_tag), len(index_to_word)])

    for para in train_data_index:
        # inital probability matrix
        init[para[0][1]] += 1
        # transfer probability matrix
        state_last = para[0][1]
        for i, word in enumerate(para):
            if i > 0:
                # transfer probability matrix
                state_now = word[1]
                trans[state_last, state_now] += 1
                state_last = state_now
            # emit probability matrix
            emit[word[1], word[0]] += 1
    trans = trans / np.sum(trans, axis=1).reshape(-1, 1)
    init = init / np.sum(init)
    emit = emit / np.sum(emit, axis=1).reshape(-1, 1)

    # save matrix file
    np.savetxt(args.hmmtrans, trans, delimiter=' ')
    np.savetxt(args.hmmemit, emit, delimiter=' ')
    np.savetxt(args.hmminit, init, delimiter=' ')
