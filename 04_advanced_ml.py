import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("1. 正在加载 AKShare 数据 (200只股票)...")

print("1. 正在加载 AKShare 数据 (200只股票)...")
all_files = glob.glob("data_ak/*.csv")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# === 智能修复列名 ===
# 如果发现列名里还有中文的 '日期'，就把它改名
if '日期' in df.columns:
    df.rename(columns={'日期': 'trade_date'}, inplace=True)
# 如果发现带有空格的 ' trade_date'，也把它清理掉
df.columns = df.columns.str.strip()

# 再次尝试转换日期
try:
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
except KeyError:
    print("还是找不到 trade_date 列，请检查你的 csv 文件表头：")
    print(df.columns)
    exit()

df['ret'] = df['pct_chg'] / 100.0
df = df.sort_values(['ts_code', 'trade_date'])



df['ret'] = df['pct_chg'] / 100.0
df = df.sort_values(['ts_code', 'trade_date'])

print("2. 正在构建特征...")
def calc_features(group):
    # 动量因子
    group['mom_5'] = group['ret'].rolling(5).sum()
    group['mom_20'] = group['ret'].rolling(20).sum()
    # 波动率因子
    group['vol_20'] = group['ret'].rolling(20).std()
    # 振幅因子
    group['amplitude'] = (group['high'] - group['low']) / group['close']
    # 换手率因子 (使用真实换手率)
    group['turnover_5'] = group['turnover'].rolling(5).mean()
    group['turnover_20'] = group['turnover'].rolling(20).mean()
    # 量价背离：换手率变化率
    group['turnover_change'] = group['turnover_5'] / group['turnover_20']
    
    # 目标变量：下一天的收益率
    group['target'] = group['ret'].shift(-1)
    return group

df = df.groupby('ts_code', group_keys=False).apply(calc_features)
df = df.dropna(subset=['mom_5', 'mom_20', 'vol_20', 'amplitude', 'turnover_5', 'turnover_20', 'turnover_change', 'target']).copy()

features = ['mom_5', 'mom_20', 'vol_20', 'amplitude', 'turnover_5', 'turnover_20', 'turnover_change']


print("3. 正在进行截面市值中性化 (防小盘股暴露)...")
# 创建一个空列表，用来存处理后的数据
neutralized_data = []

# 获取所有交易日
all_dates = df['trade_date'].unique()

# 按天循环处理 (虽然慢一点，但绝对不会丢列，且逻辑最清晰)
for date in all_dates:
    daily_data = df[df['trade_date'] == date].copy()
    if len(daily_data) < 20: 
        continue
        
    daily_data = daily_data.dropna(subset=['total_mv'])
    if len(daily_data) < 20:
        continue
        
    log_mv = np.log(daily_data['total_mv'] + 1)
    X = sm.add_constant(log_mv)
    
    # 对每个因子做回归取残差
    for f in features:
        # 把 NaN 填充为 0 防止报错
        daily_data[f] = daily_data[f].fillna(0)
        res = sm.OLS(daily_data[f], X).fit()
        daily_data[f] = res.resid
        
    neutralized_data.append(daily_data)

# 重新拼成一个大表
df = pd.concat(neutralized_data, ignore_index=True)
print(f"中性化完成，处理后的数据量: {len(df)} 行")

print("4. 正在按时间切分训练集(2018-2021)与测试集(2022-2023)...")

# === 最关键的一步：强行把日期转回 datetime，如果转不了就用字符串比较 ===
try:
    df['trade_date'] = pd.to_datetime(df['trade_date'])
except:
    pass

# 这里为了绝对安全，我们用 pd.to_datetime 进行时间比较
train_df = df[df['trade_date'] < pd.to_datetime('2022-01-01')]
test_df = df[df['trade_date'] >= pd.to_datetime('2022-01-01')]

if len(train_df) == 0 or len(test_df) == 0:
    print("时间切分失败！检查你的 trade_date 列到底是什么格式：")
    print(df['trade_date'].head())
    exit()

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']



print("5. 正在训练 LightGBM 模型...")
params = {'objective': 'regression', 'metric': 'rmse', 'max_depth': 3, 'learning_rate': 0.03, 'num_leaves': 15, 'verbose': -1}
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[train_data, valid_data], callbacks=[lgb.early_stopping(30, verbose=False)])

# 打印特征重要性
importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importance(importance_type='gain')}).sort_values(by='Importance', ascending=False)
print("\n========== 因子重要性排序 ==========\n", importance.to_string(index=False))

print("\n6. 正在运行回测 (已扣除双边千二交易成本)...")
test_df['pred'] = model.predict(X_test)
daily_returns = []
dates = test_df['trade_date'].unique()

for date in dates:
    daily_data = test_df[test_df['trade_date'] == date].copy()
    if len(daily_data) < 50: continue
    
    # 过滤涨跌停
    daily_data = daily_data[(daily_data['pct_chg'] < 9.5) & (daily_data['pct_chg'] > -9.5)]
    if len(daily_data) < 40: continue
    
    # 每天买预测分最高的 20 只，做空预测分最低的 20 只（扩大股票数量减少波动）
    daily_data = daily_data.sort_values('pred')
    long_stocks = daily_data.tail(20)
    short_stocks = daily_data.head(20)
    
    long_ret = long_stocks['target'].mean()
    short_ret = short_stocks['target'].mean()
    
    # 扣除双边千二交易成本 (0.002)
    trading_cost = 0.002 
    strategy_ret = long_ret - short_ret - trading_cost
    
    daily_returns.append({'date': date, 'return': strategy_ret})

res_df = pd.DataFrame(daily_returns).set_index('date')
res_df['nav'] = (1 + res_df['return']).cumprod()

annual_ret = res_df['return'].mean() * 252
annual_vol = res_df['return'].std() * np.sqrt(252)
sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
max_dd = ((res_df['nav'] - res_df['nav'].cummax()) / res_df['nav'].cummax()).min()

print("\n========== 进阶版策略业绩 (测试集 2022-2023) ==========")
print(f"年化收益: {annual_ret*100:.2f}%")
print(f"年化波动: {annual_vol*100:.2f}%")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print("======================================================\n")

plt.figure(figsize=(10, 5))
plt.plot(res_df.index, res_df['nav'], color='purple', label='Advanced LightGBM')
plt.axhline(y=1, color='gray', linestyle='--')
plt.title('Advanced Multi-Factor Strategy (Neutralized + Cost)')
plt.legend()
plt.grid(True)
plt.show()