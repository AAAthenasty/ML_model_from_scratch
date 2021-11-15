# -*- coding: UTF-8 -*-
"""
@Project ：10601 
@File    ：plot_graph.py
@Author  ：Tianye Song
@Date    ：11/13/21 18:17:09 
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'# Sequences': ['10', '100', '1000', '10000'],
              "Train Average Log-Likelihood": [-97.9105, -78.6899, -64.6742, -60.6235],
              "Validation Average Log-Likelihood":[-87.9715, -80.8322, -70.4556, -61.0856]})
plt.plot(df['# Sequences'], df["Train Average Log-Likelihood"], label='train')
plt.plot(df['# Sequences'], df["Validation Average Log-Likelihood"], label='validation')
plt.legend()
plt.title("Log-Likelihood")
plt.show()

