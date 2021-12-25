# -*- coding: UTF-8 -*-
"""
@Project ：inspection.py 
@File    ：plot.py
@Author  ：Tianye Song
@Date    ：11/22/21 23:02:18 
"""

import pandas as pd
import matplotlib.pyplot as plt
df_raw = pd.read_csv("tile_returns.txt", names=['value'])

plt.plot(df_raw.value, label='ori')
plt.plot(df_raw.rolling(25).mean().value, label='rolling')
plt.title("Reward of tile method ")
plt.legend()
plt.show()
