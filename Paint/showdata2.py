# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from matplotlib import font_manager
from itertools import accumulate

from collections import Counter

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取数据
df = pd.read_csv('../data_single.csv')
df = df.groupby('label')

df01 = df.get_group("正面")["evaluation"]
df02 = df.get_group("负面")["evaluation"].tolist()

# 分词并统计
# 正面
jiebaList01 = []

for i in range(len(df01.index)):
    jiebaList01.extend(jieba.lcut(df01[i]))
result01 = Counter(jiebaList01)

del result01["，"]
del result01["的"]
del result01["了"]
del result01["。"]
del result01["！"]
del result01[" "]
del result01["是"]
del result01["还"]
del result01["我"]
del result01["也"]
del result01["很"]
del result01["就"]
del result01["用"]
del result01["给"]
del result01["在"]
del result01["："]
del result01[":"]
del result01[","]
del result01["."]
del result01["。"]

most01 = result01.most_common(30)
mostX01 = []
mostY01 = []
for i in range(len(most01)):
    mostX01.append(most01[i][0])
    mostY01.append(most01[i][1])

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x=mostX01, height=mostY01)
ax.set_title("正面高频词汇图", fontsize=15)
for a, b in zip(mostX01, mostY01):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
plt.savefig("../Picture/正面高频词云图.png")

plt.show()

# 反面
jiebaList02 = []

for i in range(len(df02)):
    jiebaList02.extend(jieba.lcut(df02[i]))
result02 = Counter(jiebaList02)

del result02["，"]
del result02["的"]
del result02["了"]
del result02["。"]
del result02["！"]
del result02[" "]
del result02["是"]
del result02["还"]
del result02["我"]
del result02["也"]
del result02["很"]
del result02["就"]
del result02["用"]
del result02["给"]
del result02["在"]
del result02["："]
del result02[":"]
del result02[","]
del result02["."]
del result02["。"]

most02 = result02.most_common(30)
print(type(most02))
mostX02 = []
mostY02 = []
for i in range(len(most02)):
    mostX02.append(most02[i][0])
    mostY02.append(most02[i][1])

fig, bx = plt.subplots(figsize=(10, 7))
bx.bar(x=mostX02, height=mostY02)
bx.set_title("负面高频词汇图", fontsize=15)
for a, b in zip(mostX02, mostY02):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
plt.savefig("../Picture/反面高频词云图.png")
plt.show()
