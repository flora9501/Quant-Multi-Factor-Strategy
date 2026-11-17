import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取刚才保存的数据
df = pd.read_csv('maotai_2023.csv')

# 2. 数据处理：把日期变成索引，并按时间正序排列（回测的大忌是时间倒序！）
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values('trade_date')
df.set_index('trade_date', inplace=True)

# 3. 计算因子
# 因子1：5日动量（过去5天的累计收益率）
# pct_chg 是百分比，所以要除以 100 变成小数
df['ret'] = df['pct_chg'] / 100.0
df['mom_5'] = df['ret'].rolling(window=5).sum()

# 因子2：20日波动率（过去20天收益率的标准差，通常乘以根号252年化）
df['vol_20'] = df['ret'].rolling(window=20).std() * np.sqrt(252)

# 4. 打印看看算出来的因子
print("带有因子的数据前 10 行：")
# 因为前 20 天没有足够数据算波动率，会有 NaN，这是正常的
print(df[['close', 'ret', 'mom_5', 'vol_20']].head(10)) 

# 5. 可视化：画出股价和 5日动量
fig, ax1 = plt.subplots(figsize=(12, 6))

# 画左轴：收盘价
ax1.plot(df.index, df['close'], color='blue', label='Close Price')
ax1.set_ylabel('Close Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 画右轴：5日动量因子
ax2 = ax1.twinx()
ax2.plot(df.index, df['mom_5'], color='red', alpha=0.6, label='5-Day Momentum')
ax2.set_ylabel('5-Day Momentum', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Maotai Close Price vs 5-Day Momentum')
plt.grid(True)
plt.show()