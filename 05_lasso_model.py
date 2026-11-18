import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

print("1. 正在加载并合并数据...")
all_files = glob.glob("data_ak/*.csv")
df_list = []
for f in all_files:
    temp_df = pd.read_csv(f)
    # 【绝对防弹】：直接从文件名提取股票代码，无视 csv 里乱七八糟的列名
    temp_df['ts_code'] = f.split('/')[-1].split('.')[0] 
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# 统一列名 (兼容可能的中文列名)
df.rename(columns={'日期': 'trade_date', '涨跌幅': 'pct_chg', '换手率': 'turnover'}, inplace=True)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
df['ret'] = df['pct_chg'] / 100.0
df = df.sort_values(['ts_code', 'trade_date'])

print("2. 正在构建特征与目标...")
def calc_features(group):
    group['mom_5'] = group['ret'].rolling(5).sum()
    group['mom_20'] = group['ret'].rolling(20).sum()
    group['vol_20'] = group['ret'].rolling(20).std()
    group['amplitude'] = (group['high'] - group['low']) / group['close']
    group['turnover_5'] = group['turnover'].rolling(5).mean()
    group['turnover_20'] = group['turnover'].rolling(20).mean()
    group['target'] = group['ret'].shift(-1)
    return group

# 计算特征
df = df.groupby('ts_code', group_keys=False).apply(calc_features)

# 【终极防坑】：groupby.apply 之后，ts_code 可能变成索引，我们强行把它复原回列！
df = df.reset_index()

features = ['mom_5', 'mom_20', 'vol_20', 'amplitude', 'turnover_5', 'turnover_20']

# 清洗异常值
df = df.dropna(subset=features + ['target'])
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features).copy()

print("3. 正在按时间切分训练集和测试集...")
train_df = df[df['trade_date'] < '2022-01-01']
test_df = df[df['trade_date'] >= '2022-01-01']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. 正在训练 Lasso 模型...")
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lasso.coef_})
print("\n========== Lasso 因子权重 ==========")
print(coef_df.to_string(index=False))
print("====================================\n")


print("5. 正在运行回测 (5天调仓，扣除千二成本)...")
test_df = test_df.copy()
test_df['pred'] = lasso.predict(X_test_scaled)

# === 终极防弹：把 DataFrame 转成最基础的 Python 字典 ===
# 避免任何 pandas 的索引和列名丢失问题
test_records = test_df.to_dict('records')

# 按日期把数据重组： { '2022-01-01': [股票1, 股票2...], ... }
from collections import defaultdict
daily_data_dict = defaultdict(list)

for row in test_records:
    # 提取日期，转为字符串确保匹配
    date_str = pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d')
    
    # 提取股票代码（这里直接取文件名切出来的，如果找不到就取原来的）
    code = row.get('ts_code', row.get('股票代码'))
    
    # 过滤涨跌停
    if row['pct_chg'] < 9.5 and row['pct_chg'] > -9.5:
        daily_data_dict[date_str].append({
            'code': code,
            'pred': row['pred'],
            'target': row['target']
        })

daily_returns = []
dates = sorted(daily_data_dict.keys())

for i in range(0, len(dates), 5):
    date = dates[i]
    hold_dates = dates[i:i+5]
    if len(hold_dates) < 5: break
    
    current_day_data = daily_data_dict[date]
    if len(current_day_data) < 50: continue
    
    # 按 pred 排序
    current_day_data = sorted(current_day_data, key=lambda x: x['pred'])
    
    # 选多头（最后20个，分最高）和空头（前20个，分最低）
    long_stocks = current_day_data[-20:]
    short_stocks = current_day_data[:20]
    
    long_codes = [s['code'] for s in long_stocks]
    short_codes = [s['code'] for s in short_stocks]
    
    long_rets, short_rets = [], []
    
    for h_date in hold_dates:
        # 构建当天的收益字典方便查找
        h_data_dict = {row['code']: row['target'] for row in daily_data_dict[h_date]}
        
        # 提取当天收益（如果停牌则用 0 填充）
        l_ret = np.mean([h_data_dict.get(code, 0) for code in long_codes])
        s_ret = np.mean([h_data_dict.get(code, 0) for code in short_codes])
        
        long_rets.append(l_ret)
        short_rets.append(s_ret)
        
    strategy_ret = np.nanmean(long_rets) - np.nanmean(short_rets) - 0.002 
    daily_returns.append({'date': date, 'return': strategy_ret})

# ================= 6. 计算业绩并画图 =================
res_df = pd.DataFrame(daily_returns)
res_df['date'] = pd.to_datetime(res_df['date'])
res_df.set_index('date', inplace=True)

res_df['nav'] = (1 + res_df['return']).cumprod()

annual_ret = res_df['return'].mean() * 252
annual_vol = res_df['return'].std() * np.sqrt(252)
sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
max_dd = ((res_df['nav'] - res_df['nav'].cummax()) / res_df['nav'].cummax()).min()

print("\n========== Lasso 策略业绩 (5天调仓) ==========")
print(f"年化收益: {annual_ret*100:.2f}%")
print(f"年化波动: {annual_vol*100:.2f}%")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print("=============================================\n")

plt.figure(figsize=(10, 5))
plt.plot(res_df.index, res_df['nav'], color='orange', linewidth=2, label='Lasso Strategy (5-Day)')
plt.axhline(y=1, color='gray', linestyle='--')
plt.title('Lasso Multi-Factor Strategy (Weekly Rebalancing)')
plt.legend()
plt.grid(True)
plt.show()