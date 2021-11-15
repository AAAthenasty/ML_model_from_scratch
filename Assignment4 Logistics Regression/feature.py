# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：feature.py
@Author  ：Tianye Song
@Date    ：10/9/21 11:16:26 
"""
import csv
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='NLP feature engineering')
parser.add_argument('train_input', type=str)
parser.add_argument('validation_input', type=str)
parser.add_argument('test_input', type=str)
parser.add_argument('dict_input', type=str)

parser.add_argument('formatted_train_out', type=str)
parser.add_argument('formatted_validation_out', type=str)
parser.add_argument('formatted_test_out', type=str)
parser.add_argument('feature_flag', type=int)
parser.add_argument('feature_dictionary_input', type=str)


def read_file(input_path, deli='\t'):
    with open(input_path) as file:
        raw = csv.reader(file, delimiter=deli)
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


def filter_text(text_list, filter_dict):
    return [x for x in text_list if x in filter_dict.keys()]


def regular_feature_engineering(text_list, text_list_dict):
    result = np.zeros([text_list.shape[0], len(text_list_dict)])
    text_dict_keys = np.array(list(text_list_dict.keys()))
    for i in np.arange(result.shape[0]):
        text_set = set(text_list[i])
        for j in np.arange(result.shape[1]):
            if text_set & {text_dict_keys[j]}:
                result[i, j] = 1
    return result


def w2v_feature_engineering(text_list, text_list_dict):
    result = np.zeros([text_list.shape[0], len(list(text_list_dict.values())[0])])
    for i in np.arange(result.shape[0]):
        count = 0
        for word in text_list[i]:
            result[i] = result[i] + text_list_dict[word]
            count = count + 1
        if count != 0:
            result[i] = result[i]/count
    return result


if __name__ == '__main__':
    #t0 = time.time()
    args = parser.parse_args()

    text_dict_file = read_file(args.dict_input)
    text_dict = dict([x.split(" ") for x in text_dict_file[:, 0]])

    w2v_file = read_file(args.feature_dictionary_input)
    w2v_dict = dict([(x[0], np.array([float(y) for y in x[1:]])) for x in w2v_file])

    args_list = ["train", "valid", "test"]
    for arg in args_list:
        if arg == "train":
            file = read_file(args.train_input)
            output_path = args.formatted_train_out
        elif arg == 'valid':
            file = read_file(args.validation_input)
            output_path = args.formatted_validation_out
        else:
            file = read_file(args.test_input)
            output_path = args.formatted_test_out
        file_tag = np.array(file[:, 0])
        file_feature = np.array([filter_text(x.split(" "), text_dict) for x in file[:, 1]], dtype=object)
        if args.feature_flag == 1:
            file_feature_engineering = regular_feature_engineering(file_feature, text_dict)
            save_fmt = "%d"
        else:
            file_feature_engineering = w2v_feature_engineering(file_feature, w2v_dict)
            save_fmt = "%.6e"
        file_feature_engineering = np.insert(file_feature_engineering, 0, values=file_tag, axis=1)
        np.savetxt(output_path, file_feature_engineering, delimiter="\t", fmt=save_fmt)
    #print(time.time()-t0)