import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 导入字典
with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)

try:
    # 数据预处理
    input_shape = 180

    # 载入模型
    model_save_path = './trained_model.h5'
    lstm_model = load_model(model_save_path)
    while (1):
        print("请输入您的评论：")
        sent = input()
        x = [[word_dictionary[word] for word in sent]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

        # 模型预测
        y_predict = lstm_model.predict(x)
        label_dict = {v: k for k, v in output_dictionary.items()}
        print('输入语句为: %s' % sent)
        print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])

except KeyError as err:
    print("您输入的句子有汉字不在词汇表中，请重新输入！")
    print("不在词汇表中的单词为：%s." % err)
