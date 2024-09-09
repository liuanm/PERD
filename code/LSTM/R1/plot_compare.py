import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 从CSV加载数据
df1 = pd.read_csv('R1.csv')
df2 = pd.read_csv('interpolated_data.csv')

# 将日期列转换为datetime类型
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])
plt.rc('font',family='Times New Roman', size=11)
#plt.figure(figsize=(6, 4 ))
plt.figure( dpi=600)
# 绘制第一个时间序列
plt.plot(df1['Date'], df1['Cum'], label='raw data')

# 绘制第二个时间序列
plt.plot(df2['Date'], df2['interpolated_Cum'], label='interpolated data')

# 设置x轴日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.subplots_adjust(left=0.12, right=0.95, top=0.98, bottom=0.11)
plt.xlabel('Date')
plt.ylabel('Cumulative deformation (mm)')
## 图序
text_date1 = datetime(2017, 11, 1)
plt.text(text_date1, -200, '(a)', fontsize=18, color='black')

# 设置图例
plt.legend(frameon=False)
plt.savefig('interpolated.png')
# 显示图形
plt.show()
