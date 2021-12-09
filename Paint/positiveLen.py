# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

plt.style.use('bmh')
df=pd.read_csv('../data_single.csv')
df=df[(df['label'] == '正面')]
df['length'] = df['evaluation'].apply(lambda x: len(x))
len_df = df.groupby('length').count()
sent_length = len_df.index.tolist()
sent_freq = len_df['evaluation'].tolist()

plt.bar(sent_length, sent_freq)
plt.title("正面句子长度及出现频数统计图")
plt.xlabel("正面句子长度")
plt.ylabel("正面句子长度出现的频数")
plt.savefig("../Picture/正面句子长度及出现频数统计图.png")

plt.show()