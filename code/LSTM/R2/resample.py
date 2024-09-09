import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取CSV文件，假设文件中有两列：'timestamp'和'value'
df = pd.read_csv('R2.csv')

# 将时间戳列转换为datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 设置目标时间间隔，例如15天一个数据点
target_interval = pd.to_timedelta('15 days')

# 创建目标时间戳序列
target_timestamps = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq=target_interval)

# 使用反距离加权法插值
def inverse_distance_weighting(ts, values, target_ts, power=2):
    weights = 1 / ((target_ts - ts).dt.total_seconds().abs()+0.000001) ** power
    weighted_sum = np.sum(values * weights)
    total_weights = np.sum(weights)
    interpolated_value = weighted_sum / total_weights
    return interpolated_value

# 对每个目标时间戳进行插值
interpolated_values = []
for target_ts in target_timestamps:
    nearest_indices = np.abs(df['Date'] - target_ts).argsort()[:2]
    nearest_values = df.loc[nearest_indices, 'Cum']
    interpolated_value = inverse_distance_weighting(df.loc[nearest_indices, 'Date'], nearest_values, target_ts)
    interpolated_values.append(interpolated_value)

# 创建新的DataFrame保存插值后的数据
interpolated_df = pd.DataFrame({'Date': target_timestamps, 'interpolated_Cum': interpolated_values})

# 输出结果或保存到文件
print(interpolated_df)

# 保存到CSV文件
interpolated_df.to_csv('interpolated_data.csv', index=False)
