import os
import random
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from numpy import sqrt
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils

import warnings

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


warnings.filterwarnings('ignore')

# 读取数据集
dataset = pd.read_csv('interpolated_data.csv')
# 显示shape
print(dataset.shape)
# 默认显示前5行
print(dataset.head())
# 显示数据描述
print(dataset.describe())
# 显示字段数据类型
print(dataset.dtypes)
# 将字段Datetime数据类型转换为日期类型
dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y/%m/%d")
# 再次查看字段的数据类型
print(dataset.dtypes)
# 将字段Date设置为索引列
# 目的：后续基于索引来进行数据集的切分
dataset.index = dataset.Date
# 显示默认前5行
print(dataset.head())
# 将原始的Datetime字段列删除
dataset.drop(columns=['Date'], axis=1, inplace=True)
# 默认显示前5行
print(dataset.head())
# 可视化显示DOM_MW的数据分布情况
dataset['interpolated_Cum'].plot(figsize=(8,4),marker='.',markersize=3, color='red',linestyle='-')
plt.savefig("orginal_dataset.jpg")
plt.close()
plt.show()
# 数据进行归一化
scaler = MinMaxScaler()
print(dataset['interpolated_Cum'])
print(dataset['interpolated_Cum'].values)
dataset['interpolated_Cum'] = scaler.fit_transform(dataset['interpolated_Cum'].values.reshape(-1, 1))  # 转化为列向量
print(dataset.head())
# 可视化显示DOM_MW的数据分布情况
dataset['interpolated_Cum'].plot(figsize=(8,4),marker='.',markersize=3, color='red',linestyle='-')
print(dataset.head())
plt.savefig("scalered_dataset.jpg")
plt.close()
# plt.show()

# 特征工程
# 功能函数：构造特征数据集和标签集
def create_new_dataset(dataset, seq_len=12):
    '''基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）

    Returns:
        X, y
    '''
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label

    # 返回特征数据集和标签集
    return np.array(X), np.array(y)


# 功能函数：基于新的特征的数据集和标签集，切分：X_train, X_test

def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例

    Returns:
        X_train, X_test, y_train, y_test
    '''
    global train_data_len
    X_len = len(X)  # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio)  # 训练集的样本数量

    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集

    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    # 返回值
    return X_train, X_test, y_train, y_test


# 功能函数：基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)

def create_batch_data(X, y, batch_size=32, data_type=1):
    '''基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）

    Returns:
        train_batch_data 或 test_batch_data
    '''
    if data_type == 1:  # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size)  # 构造批数据
        # 返回
        return test_batch_data
    else:  # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        train_batch_data = dataset.cache().shuffle(50).batch(batch_size)  # 构造批数据
        # 返回
        return train_batch_data


# ① 原始数据集
dataset_original = dataset
print("原始数据集: ", dataset_original.shape)

# ② 构造特征数据集和标签集，seq_len序列长度为12小时
SEQ_LEN = 2 # 序列长度
X, y = create_new_dataset(dataset_original.values, seq_len = SEQ_LEN)
print(X.shape)
print(y.shape)
# 样本1 - 特征数据
print(X[0])
print(y[0])
# ③ 数据集切分
X_train, X_test, y_train, y_test = split_dataset(X, y, train_ratio=0.85)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train[0:3])
print(y_train[0:3])
print(X_test[0:3])
print(y_test[0:3])


# 设置随机种子
def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_tensorflow(1)


# ④ 基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
# 测试批数据
test_batch_dataset = create_batch_data(X_test, y_test, batch_size=5, data_type=1)  # 将数据集中的连续 batch_size 个元素组成一个批次
# 训练批数据
train_batch_dataset = create_batch_data(X_train, y_train, batch_size=5, data_type=2)


# 构建模型
def create_model(units=50, learning_rate=0.0001, l2_penalty=0.1):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(SEQ_LEN,1), return_sequences=False,kernel_regularizer=l2(l2_penalty)))
    # model.add(Dropout(0.5))
    ##未归一化会导致训练集后期效果变差，模型过于复杂也会导致效果差
    # model.add(LSTM(6, return_sequences=False,kernel_regularizer=l2(l2_penalty)))
    # model.add(Dropout(0.1))
    # model.add(LSTM(60, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(10, return_sequences=False, kernel_regularizer=l2(l2_penalty)))
    # model.add(Dropout(0.1))
    # model.add(LSTM(30, return_sequences=False))
    # model.add(Dropout(0.1))

    model.add(Dense(1))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    # model.compile(optimizer='adam', loss="mse")
    # model.compile(loss='mse')

    return model

# 创建一个KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=1)

# 定义参数网格
parameters = {
    'batch_size': [1, 2, 3, 4, 5],
     'units': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

# 创建GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid_search = grid_search.fit(X_train, y_train, validation_data=(X_test, y_test))

# 获取最佳参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 访问最佳模型
best_model = grid_search.best_estimator_

# 定义 checkpoint，保存权重文件
file_path = "best_checkpoint.hdf5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)

early_stopping_callback = EarlyStopping(monitor='loss', patience=5)
callbacks=[
    early_stopping_callback,
    checkpoint_callback
]

# best_model=create_model()
# 模型训练
history = best_model.fit(
    X_train,
    y_train,
    epochs=500,
    validation_data=(X_test,y_test),
    callbacks=callbacks)

# 保存最佳模型
best_model.model.save('lstm_model')

# 显示 train loss 和 val loss
plt.figure(figsize=(8,4))
plt.rc('font',family='Times New Roman', size=11)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.11)
plt.savefig("loss.jpg")
plt.show()

# 训练集
train_pred = best_model.predict(X_train,verbose=1)
train_pred_true = scaler.inverse_transform(train_pred.reshape(-1,1))
y_train_true = scaler.inverse_transform(y_train.reshape(-1,1))
x_date=dataset.index[SEQ_LEN:train_data_len+SEQ_LEN]
# plt.figure(figsize=(8, 4))
# year_starts = [datetime(year, 1, 1) for year in range(x_date[0].year, x_date[-1].year + 1)]
# plt.plot(x_date,y_train_true, label="True_train", marker='^',color='b',linestyle='-')
# plt.plot(x_date,train_pred_true, label="pred_train",marker='*',color='r',linestyle='-')
# plt.xticks(year_starts, [year.strftime('%Y') for year in year_starts])
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#
# plt.title("True_Pred_train")
# plt.savefig("True_Pred_train.jpg")


# 计算r2
# score = r2_score(y_test_true, test_pred_true)
test_pred = best_model.predict(X_test, verbose=1)
test_pred_true = scaler.inverse_transform(test_pred.reshape(-1,1))
print(test_pred.shape)
print(y_test.shape)
print(test_pred)
y_test_true = scaler.inverse_transform(y_test.reshape(-1,1))
# rmse = sqrt(mean_squared_error(y_test, test_pred))
rmse = sqrt(mean_squared_error(y_test_true, test_pred_true))
print("rmse的值：", rmse)
# plt.figure(figsize=(8,4))
# # 绘制模型验证集结果
# plt.plot(y_test, label="True", marker='^',color='b',linestyle='-')
# plt.plot(test_pred, label="pred",marker='*',color='r',linestyle='-')
# plt.title("True vs Pred")
# plt.show()
# plt.savefig("True_Pred_scaled.jpg")

# plt.figure(figsize=(8,4))
# plt.plot(y_test_true, label="True label",marker='^',color='b',linestyle='-')
# plt.plot(test_pred_true, label="Pred label",marker='*',color='r',linestyle='-')
# plt.title("True vs Pred")
# plt.legend(loc='best')
# plt.savefig("True_Pred.jpg")
# plt.show()

# 选择test中的最后一个样本
sample = X_test[-1]
# print(sample.shape)
sample = sample.reshape(1, sample.shape[0], 1)
# print(sample.shape)

# 模型预测
sample_pred = best_model.predict(sample)
# print(sample_pred)
ture_data = X_test[-1] # 真实test的最后20个数据点
# print(ture_data)
# print(ture_data.shape)
# print(list(ture_data[:,0]))

def predict_next(model, sample, epoch=10):  # 预测未来20个值
    temp1 = list(sample[:,0])
    for i in range(epoch):
        sample = sample.reshape(1, SEQ_LEN, 1)   # batch_size 是样本的数量，input_dim 是每个样本的特征维度。
        pred = model.predict(sample)
        # value = pred.tolist()[0][0]
        temp1.append(pred)
        sample = np.array(temp1[i+1 : i+SEQ_LEN+1])
    return temp1
preds = predict_next(best_model, ture_data, 12)[SEQ_LEN:]
preds = np.array(preds)
preds = preds.reshape(-1, 1)
preds_true = scaler.inverse_transform(preds.reshape(-1,1))
# plt.figure(figsize=(8,4))
# plt.plot(preds, color='red', label='Prediction',marker='*')
# #plt.plot(ture_data, color='blue', label='Truth',marker='*')
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.legend(loc='best')
# plt.savefig("predict.jpg")
# plt.show()
##绘制总图
y_all_true=np.concatenate((y_train_true, y_test_true), axis=0)
# all_pred_true=np.concatenate((train_pred,test_pred,preds),axis=0)
all_pred_true=np.concatenate((train_pred_true,test_pred_true,preds_true),axis=0)

x_all_date=dataset.index[SEQ_LEN:]
interval = 15
num_new_dates = 12


for i in range(num_new_dates):
    new_date = dataset.index[-1] + timedelta(days=interval)

    dataset = dataset.append(pd.DataFrame(index=[new_date]))


pred_date=dataset.index[SEQ_LEN:]
plt.figure(dpi=600)
year_starts = [datetime(year, 1, 1) for year in range(x_all_date[0].year, pred_date[-1].year + 1)]
plt.rc('font',family='Times New Roman', size=11)
plt.plot(x_all_date, y_all_true, label="InSAR monitoring data", color='black', linestyle='-',linewidth=1.0)
plt.xticks(year_starts, [year.strftime('%Y') for year in year_starts])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.plot(pred_date[:train_data_len+1], all_pred_true[:train_data_len+1], label="training data", color='b', linestyle='-',linewidth=1.0)
plt.plot(pred_date[train_data_len:-num_new_dates], all_pred_true[train_data_len:-num_new_dates], label="Test data", color='r', linestyle='-',linewidth=1.0)
plt.plot(pred_date[-num_new_dates-1:],all_pred_true[-num_new_dates-1:], label="Prediction data",color='lawngreen',linestyle='-',linewidth=1.0)
R_to_display = f'Test RMSE: {rmse:.2f}'
text_date1 = datetime(2021, 12, 1)
text_date2 = datetime(2017, 12, 1)
plt.text(text_date1, -50, R_to_display, color='red')
# plt.text(text_date2, -5, '(a)', fontsize='medium', color='black')
plt.xlabel('Date')
plt.ylabel('Cumulative deformation (mm)')
plt.legend(frameon=False)
plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.1)
plt.savefig('lstm_all.png')
plt.show()