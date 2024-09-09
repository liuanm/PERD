import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
# 读取数据
data = pd.read_csv('R1.csv')

# 定义拟合函数
def model(t, S, lmbda, k):
        denominator = np.power(k, lmbda)+np.power(t, lmbda)
        return S * np.power(t, lmbda) / denominator

# 将日期转换为距离2018年1月10日的天数
date_format = '%Y-%m-%d' # 日期格式
start_date = pd.to_datetime('2018-01-10', format=date_format)
data['Date'] = pd.to_datetime(data['Date'], format=date_format)
# print(data.head())
data['Days'] = (data['Date'] - start_date).dt.days

# data['Days'] = (pd.to_datetime(data['Date'], format=date_format) - start_date).dt.days

# 提取数据

t = data['Days'].values[:130]
t_y = data['Days'].values[130:]
L = data['Cum'].values[:130]
L_y = data['Cum'].values[130:]
# 进行最小二乘拟合
popt, pcov = curve_fit(model, t, L, bounds=([-np.inf, -np.inf, 1e-06], [0, np.inf, np.inf]),maxfev=10000)

# 输出拟合参数
S_opt, lambda_opt, k_opt = popt
print("Optimized parameters:")
print("S =", S_opt)
print("lambda =", lambda_opt)
print("k =", k_opt)
u=data['Cum'].values[-1]/S_opt
print("U=", u)
# 计算拟合曲线的一阶导数
def model_derivative(t, S, lmbda, k):
    first_derivative = S * lmbda * np.power(t, lmbda - 1) / (np.power(k, lmbda) + np.power(t, lmbda)) - S * np.power(t, lmbda) * lmbda * np.power(t, lmbda - 1) / (
                               np.power(k, lmbda) + np.power(t, lmbda)) ** 2
    return first_derivative

future_dates = pd.date_range(start='2023-01-08', end='2050-01-01')
# 将未来日期转换为距离2018年1月10日的天数
future_days = (future_dates - start_date).days

# 使用拟合模型计算未来日期的累积结算
future_cumulative = model(future_days, *popt)
#计算训练数据的rmse
L_model = model(t,*popt)
rmse = sqrt(mean_squared_error(L, L_model))
print("训练集rmse的值：", rmse)
T_to_display = f'Training RMSE: {rmse:.2f}'
# 寻找使得拟合函数的一阶导数输出值接近1的日期
def find_nearest_date(target_value, start_date, model, popt):
    # 定义目标值
    target = target_value
    # 定义日期范围
    search_dates = pd.date_range(start=start_date, end='2050-01-01')
    # 转换为距离2018年1月10日的天数
    search_days = (search_dates - start_date).days[5:]
    # 计算拟合函数输出值
    predictions = model(search_days, *popt)
    # 找到最接近目标值的日期
    nearest_date_index = np.argmin(np.abs(predictions - target))
    nearest_date = search_dates[nearest_date_index]
    return nearest_date

# 寻找使得拟合函数输出值接近1/365的日期
target_value = -277
nearest_date = find_nearest_date(target_value, start_date, model, popt)
end_date=nearest_date.date()
print("The date when dL(t)/dt is closest to -1 is:", end_date)
nearest_days = (nearest_date - start_date).days
der_end = model_derivative(nearest_days, *popt)*365
# 绘制数据点
plt.rc('font',family='Times New Roman', size=11)
fig, axes = plt.subplots(2, 1,dpi=600,figsize=(8, 7))
axes[0].scatter(data['Date'][:130], L, label='training data',color='blue',s=8)
axes[0].scatter(data['Date'][130:],L_y,label='test data',color='red',s=8)
# 绘制拟合曲线
t_fit = np.linspace(min(t), max(t), 100)  # 生成拟合曲线上的点
t_fit_dates = [start_date + timedelta(days=int(day)) for day in t_fit]  # 将天数加到开始日期上得到日期列表
L_fit = model(t_fit, *popt)
# 2018年1月10日——2022年12月27日
axes[0].plot(t_fit_dates, L_fit, color='black', label='Fitted Curve') # 计算拟合曲线上的值
# 2023年1月8日-2060年1月1日
axes[0].plot(future_dates, future_cumulative, 'lime', label='preditive data', zorder=1)
axes[0].axhline(y=target_value, color='red', linestyle='--', linewidth=1.5,zorder=2)
axes[0].scatter(nearest_date, target_value, color='blue', s=10, zorder=2)
D_to_display = f' {end_date}'
axes[0].text(nearest_date-timedelta(days=365*2), target_value+20, D_to_display, color='blue')
text_date1 = datetime(2044, 1, 1)
axes[0].text(text_date1, -113, T_to_display, color='blue')
text_date3 = datetime(2015, 6, 1)
#axes[0].text(text_date3,-10, '(a)', color='black', fontsize=12)
axes[0].set_ylabel('Accumulative Deformation (mm)')
axes[0].legend()
axes[0].legend(frameon=False)

# 计算拟合曲线的一阶导数值
L_derivative = model_derivative(t_fit, *popt)*365
future_derivative = model_derivative(future_days, *popt)*365
axes[1].plot(t_fit_dates[10:], L_derivative[10:], color='black', zorder=1)
axes[1].plot(future_dates, future_derivative, color='black', zorder=1)
axes[1].axhline(y=der_end, color='red', linestyle='--', linewidth=1.5 ,zorder=1)
axes[1].scatter(nearest_date, der_end, color='blue', s=10, zorder=2)
D_to_display3 = f' velocity: {der_end:.2f} '
axes[1].text(nearest_date-timedelta(days=365*2), -8, D_to_display3, color='blue')
#axes[1].text(text_date3,-2, '(b)', color='black', fontsize=12)
axes[1].set_ylabel('Deformation Rate (mm/year)')
axes[1].set_xlabel('Date')
axes[1].legend(frameon=False)
plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.06)
plt.savefig('rate.png')
#plt.show()

# 绘制验证集拟合图
fig, ax1 = plt.subplots(dpi=600,figsize=(10, 5))
plt.rc('font',family='Times New Roman', size=11)
ax1.scatter(data['Date'][130:],L_y,label='test data',color='red',s=8)
test_dates = pd.date_range(start='2023-01-08', end='2023-11-16')
# 将未来日期转换为距离2018年1月10日的天数
test_days = (test_dates - start_date).days
#计算验证数据的rmse
L_y_model = model(t_y,*popt)
rmse_y = sqrt(mean_squared_error(L_y, L_y_model))
print("rmse的值：", rmse_y)
R_to_display = f'Test RMSE: {rmse_y:.2f}'
test_cumulative = model(test_days, *popt)

ax1.plot(test_dates, test_cumulative, 'black', label='predictive models')
ax1.set_xlabel('Date')
ax1.set_ylabel('Accumulative Deformation (mm)')
text_date2 = datetime(2023, 9, 20)
ax1.text(text_date2, -176, R_to_display, color='blue', fontsize=12)
text_date4 = datetime(2022, 12, 25)
#ax1.text(text_date4,-83.8, '(a)', color='black', fontsize=16)
ax1.legend(frameon=False, fontsize=12, markerscale=2)
diff_values = np.abs(L_y-L_y_model)
ax2 = ax1.twinx()
L_y_dates = [start_date + timedelta(days=int(day)) for day in t_y]
ax2.bar(L_y_dates, diff_values, color='black', alpha=1, width=8)
ax2.set_ylabel('Difference (mm)')
ax2.legend(frameon=False)
ax2.set_ylim(0,50)
plt.subplots_adjust(left=0.08, right=0.95, top=0.98, bottom=0.1)
plt.savefig('test.png')
#plt.show()

