# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import accumulate

# 统计句子长度及长度出现的频数
df = pd.read_csv('../data_single.csv')
print(df.groupby('label')['label'].count())

df['length'] = df['evaluation'].apply(lambda x: len(x))
len_df = df.groupby('length').count()
sent_length = len_df.index.tolist()
sent_freq = len_df['evaluation'].tolist()

# 绘制句子长度及出现频数统计图
plt.bar(sent_length, sent_freq)
plt.title("句子长度及出现频数统计图")
plt.xlabel("句子长度")
plt.ylabel("句子长度出现的频数")
plt.savefig("../Picture/句子长度及出现频数统计图.png")
plt.show()
plt.close()

# 绘制句子长度累积分布函数(CDF)
sent_pentage_list = [(count / sum(sent_freq)) for count in accumulate(sent_freq)]

# 绘制CDF
plt.plot(sent_length, sent_pentage_list)

# 寻找分位点为quantile的句子长度
quantile = 0.81
# print(list(sent_pentage_list))
for length, per in zip(sent_length, sent_pentage_list):
    if round(per, 2) == quantile:
        index = length
        break
print("\n分位点为%s的句子长度:%d." % (quantile, index))

# 绘制句子长度累积分布函数图
plt.plot(sent_length, sent_pentage_list)
plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
plt.text(0, quantile, str(quantile))
plt.text(index, 0, str(index))
plt.title("句子长度累积分布图")
plt.xlabel("句子长度")
plt.ylabel("句子长度累积频率")
plt.savefig("../Picture/句子长度累积分布图.png")
plt.show()
plt.close()
