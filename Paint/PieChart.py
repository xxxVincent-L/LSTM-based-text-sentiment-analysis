# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('bmh')
df = pd.read_csv('../data_single.csv')
l = ['正面', '负面']
x = df['evaluation']
y = df['label']
temp = df.groupby('label').count()
a = temp['evaluation'].to_list()

plt.pie(a, labels=l, autopct='%2.1f%%')
plt.title("正负样本饼图")

plt.savefig("../Picture/正负样本饼图.png")
plt.show()
