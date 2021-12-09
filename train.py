import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


# 导入数据
# 文件的数据中，特征为evaluation, 类别为label.

def load_data(dataset_filepath, input_shape=20):
    df = pd.read_csv(dataset_filepath)

    # 标签及词汇表
    labels, vocabulary = list(df['label'].unique()), list(df['evaluation'].unique())

    # 构造字符级别的特征
    string = ''
    for word in vocabulary:
        string += word

    # print(string)
    # print("=====================")
    # 放在集合中去重
    vocabulary = set(string)
    # print(vocabulary)
    # print("========================")
    # 字典列表（将列表生成字典）
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    print(word_dictionary)
    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}

    label_dictionary = {label: i for i, label in enumerate(labels)}
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    # 用pickle序列化存入内存文件
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)

    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)

    vocab_size = len(word_dictionary.keys())  # 词汇表大小
    print(vocab_size)
    label_size = len(label_dictionary.keys())  # 标签类别数量
    print(label_size)
    # print(label_dictionary.keys())

    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['evaluation']]

    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    # print("-----------------")
    # print(x)
    # 这输出1/0
    y = [[label_dictionary[sent]] for sent in df['label']]
    # print(len(y))
    y = [to_categorical(label, num_classes=label_size) for label in y]
    # print(y)
    y = np.array([list(_[0]) for _ in y])

    # print(x.shape[0])
    # print(x.shape[1])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


# 创建深度学习模型， Embedding + LSTM + Softmax.
def LSTM_Model(n_units, input_shape, output_dim, dataset_filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(dataset_filepath)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))

    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='Picture/model_lstm.png', show_shapes=True)
    model.summary()

    return model


# 模型训练
def train(input_shape, dataset_filepath, model_save_path):
    # 模型输入参数，需要自己根据需要调整
    n_units = 100
    batch_size = 32
    epochs = 10
    output_dim = 20

    # 将数据集分为训练集和测试集

    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(dataset_filepath, input_shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)

    # 模型训练
    lstm_model = LSTM_Model(n_units, input_shape, output_dim, dataset_filepath)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    lstm_model.save(model_save_path)


def test(input_shape, dataset_filepath, model_save_path):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(dataset_filepath, input_shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)
    lstm_model = load_model(model_save_path)
    N = test_x.shape[0]  # 测试的条数
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        y_predict = lstm_model.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]

        predict.append(label_predict)
        label.append(label_true)

    acc = accuracy_score(predict, label)  # 预测准确率
    print('模型在测试集上的准确率为: %s.' % acc)


if __name__ == '__main__':
    dataset_filepath = './dataset.txt'
    model_save_path = './trained_model.h5'

    input_shape = 180
    train(input_shape, dataset_filepath, model_save_path)
    test(input_shape, dataset_filepath, model_save_path)
