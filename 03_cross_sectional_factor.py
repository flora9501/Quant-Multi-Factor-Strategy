import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import mstats

# ================= 1. 合并所有数据 =================
print("正在读取并合并 data 文件夹下的所有股票数据...")
all_files = glob.glob("data/*.csv")
df_list = []
for file in all_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
print(f"数据合并完成！总数据量：{len(df)} 行")

# 预处理
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values(['ts_code', 'trade_date'])

# ================= 2. 计算因子 =================
print("正在计算动量因子...")
# 算过去 5 天的收益率累加 (动量)
df['mom_5'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.rolling(5).sum())

# 算未来 1 天的收益率 (作为预测目标)
df['target_ret'] = df.groupby('ts_code')['pct_chg'].transform(lambda x: x.shift(-1))

# 删掉缺失值
df = df.dropna(subset=['mom_5', 'target_ret']).copy()

# ================= 3. 截面回测逻辑 (优化版) =================
daily_returns = []
dates = df['trade_date'].unique()

print("正在运行截面多空回测 (已加入去极值、标准化、涨跌停过滤)...")
for date in dates:
    daily_data = df[df['trade_date'] == date].copy()
    
    if len(daily_data) < 30: 
        continue
        
    # 【防坑大招 1：剔除涨跌停】
    # pct_chg 是当天的涨跌幅，如果超过 9.5% 或小于 -9.5%，说明大概率涨跌停了，实盘无法交易，必须剔除！
    daily_data = daily_data[(daily_data['pct_chg'] < 9.5) & (daily_data['pct_chg'] > -9.5)]
    
    if len(daily_data) < 20: 
        continue

    # 【防坑大招 2：去极值 (Winsorize)】
    # 把因子序列中超出 1% 和 99% 分位数的极端值，强行拉回边界，防止异常数据干扰排名
    daily_data['mom_5_clean'] = mstats.winsorize(daily_data['mom_5'], limits=[0.01, 0.01])
    
    # 【防坑大招 3：Z-score 标准化】
    # 让当天的因子均值为 0，标准差为 1，方便不同时期的数据进行横向对比
    daily_data['mom_5_z'] = (daily_data['mom_5_clean'] - daily_data['mom_5_clean'].mean()) / daily_data['mom_5_clean'].std()
    
    # 按标准化后的因子从小到大排序
    daily_data = daily_data.sort_values('mom_5_z')
    
    # 选出因子值最差的 10 只做多 (反转)，最好的 10 只做空
    long_stocks = daily_data.head(10) 
    short_stocks = daily_data.tail(10)
    
    # 算当天的多空收益
    long_ret = long_stocks['target_ret'].mean() / 100.0
    short_ret = short_stocks['target_ret'].mean() / 100.0
    strategy_ret = long_ret - short_ret 
    
    daily_returns.append({'date': date, 'return': strategy_ret})

# ================= 4. 计算业绩指标并画图 =================
res_df = pd.DataFrame(daily_returns).set_index('date')
res_df['nav'] = (1 + res_df['return']).cumprod() # 净值曲线

# 绩效指标计算
annual_ret = res_df['return'].mean() * 252
annual_vol = res_df['return'].std() * np.sqrt(252)
sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
cum_max = res_df['nav'].cummax()
drawdown = (res_df['nav'] - cum_max) / cum_max
max_dd = drawdown.min()

print("\n========== 5日动量反转策略 (优化后) ==========")
print(f"总交易天数: {len(res_df)} 天")
print(f"年化收益: {annual_ret*100:.2f}%")
print(f"年化波动: {annual_vol*100:.2f}%")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print("=======================================\n")

# 画出净值曲线
plt.figure(figsize=(10, 5))
plt.plot(res_df.index, res_df['nav'], label='Long-Short Strategy NAV', color='green', linewidth=2)
plt.axhline(y=1, color='gray', linestyle='--')
plt.title('5-Day Momentum Long-Short Strategy (Neutralized)')
plt.xlabel('Date')
plt.ylabel('Net Asset Value (NAV)')
plt.legend()
plt.grid(True)
plt.show()