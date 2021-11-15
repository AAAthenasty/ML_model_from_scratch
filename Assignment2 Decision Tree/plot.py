# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：plot_lr.py
@Author  ：Tianye Song
@Date    ：9/18/21 15:51:40 
"""


import pandas as pd
import matplotlib.pyplot as plt
from decisionTree import TreeNode, cal_error, read_file
import argparse

parser = argparse.ArgumentParser(description='Decision Tree Stump')
parser.add_argument('train_input', type=str)
parser.add_argument('test_input', type=str)



args = parser.parse_args()
train_feature, train_data = read_file(args.train_input)
test_feature, test_data = read_file(args.test_input)
error_list = []
for depth in range(len(train_feature)):
    stump = TreeNode(depth)
    stump.train(train_data, train_feature)
    train_predict = stump.predict(train_data)
    test_predict = stump.predict(test_data)

    train_error = cal_error(train_data[:, -1], train_predict)
    test_error = cal_error(test_data[:, -1], test_predict)

    error_list.append([depth, train_error, test_error])

error_df = pd.DataFrame(error_list, columns=['depth', 'train_error', 'test_error'])
plt.plot(error_df['depth'].values, error_df['train_error'].values, label='train_error')
plt.plot(error_df['depth'].values, error_df['test_error'].values, label='test_error')
plt.legend()
plt.title("Politicians")
plt.xlabel("depth")
plt.ylabel("error_rate")
plt.show()