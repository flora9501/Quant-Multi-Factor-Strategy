import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ================= 1. 读取并合并数据 =================
print("正在加载并合并数据...")
all_files = glob.glob("data/*.csv")
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values(['ts_code', 'trade_date'])

# ================= 2. 特征工程 (构建多因子) =================
print("正在构建特征...")
# 将所有收益率转为小数
df['ret'] = df['pct_chg'] / 100.0

# 按照股票分组计算特征
def calc_features(group):
    # 1. 动量因子：过去 5 天的累计收益率
    group['mom_5'] = group['ret'].rolling(5).sum()
    # 2. 波动率因子：过去 20 天收益率的标准差
    group['vol_20'] = group['ret'].rolling(20).std()
    # 3. 振幅因子：当日 (High - Low) / Close
    group['amplitude'] = (group['high'] - group['low']) / group['close']
    # 4. 量价相关性：过去 5 天 成交量与收益率的相关系数 (捕捉量价背离)
    group['vol_price_corr'] = group['vol'].rolling(5).corr(group['ret'])
    # 5. 昨日收益率
    group['ret_lag1'] = group['ret'].shift(1)
    
    # 目标变量 (Label)：预测明天的收益率 (注意：防未来函数必须 shift(-1))
    group['target'] = group['ret'].shift(-1)
    return group

# 应用特征计算
df = df.groupby('ts_code', group_keys=False).apply(calc_features)

# 剔除含有 NaN 的行 (因为有 rolling 窗口)
df = df.dropna(subset=['mom_5', 'vol_20', 'amplitude', 'vol_price_corr', 'ret_lag1', 'target']).copy()

# 定义特征列
features = ['mom_5', 'vol_20', 'amplitude', 'vol_price_corr', 'ret_lag1']

# ================= 3. 机器学习时序切分 (防泄漏) =================
# 绝不能用 train_test_split 随机打乱，必须按时间切分！
print("正在按时间切分训练集和测试集...")
train_df = df[df['trade_date'] < '2022-01-01']
test_df = df[df['trade_date'] >= '2022-01-01']

X_train, y_train = train_df[features], train_df['target']
X_test, y_test = test_df[features], test_df['target']

print(f"训练集 (2018-2021) 样本量: {len(X_train)}")
print(f"测试集 (2022-2023) 样本量: {len(X_test)}")

# ================= 4. 训练 LightGBM 模型 =================
print("正在训练 LightGBM 模型...")
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 4,          # 限制树深，防止过拟合
    'learning_rate': 0.05,
    'num_leaves': 15,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 训练并加入早停机制 (Early Stopping)
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# ================= 5. 特征重要性分析 (面试必考) =================
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)

print("\n========== 因子重要性排序 ==========")
print(importance.to_string(index=False))
print("====================================\n")

# ================= 6. 模型预测与回测 =================
print("正在运行机器学习多空回测...")
# 用模型预测测试集
test_df['pred'] = model.predict(X_test)

daily_returns = []
dates = test_df['trade_date'].unique()

for date in dates:
    daily_data = test_df[test_df['trade_date'] == date].copy()
    if len(daily_data) < 30: continue
    
    # 剔除涨跌停股票，防止实盘买不进卖不出
    daily_data = daily_data[(daily_data['pct_chg'] < 9.5) & (daily_data['pct_chg'] > -9.5)]
    if len(daily_data) < 20: continue
    
    # 按照预测分数排序
    daily_data = daily_data.sort_values('pred')
    
    # 选预测最差的 10 只做多（预期它反转），最好的 10 只做空
    long_stocks = daily_data.head(10)
    short_stocks = daily_data.tail(10)
    
    long_ret = long_stocks['target'].mean()
    short_ret = short_stocks['target'].mean()
    
    # 多空收益
    strategy_ret = long_ret - short_ret
    daily_returns.append({'date': date, 'return': strategy_ret})

# ================= 7. 计算业绩与画图 =================
res_df = pd.DataFrame(daily_returns).set_index('date')
res_df['nav'] = (1 + res_df['return']).cumprod()

annual_ret = res_df['return'].mean() * 252
annual_vol = res_df['return'].std() * np.sqrt(252)
sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
cum_max = res_df['nav'].cummax()
drawdown = (res_df['nav'] - cum_max) / cum_max
max_dd = drawdown.min()

print("\n========== LightGBM 多因子策略 (测试集 2022-2023) ==========")
print(f"年化收益: {annual_ret*100:.2f}%")
print(f"年化波动: {annual_vol*100:.2f}%")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print("=========================================================\n")

# 画出测试集净值曲线
plt.figure(figsize=(10, 5))
plt.plot(res_df.index, res_df['nav'], label='LightGBM Strategy NAV', color='red', linewidth=2)
plt.axhline(y=1, color='gray', linestyle='--')
plt.title('LightGBM Multi-Factor Strategy (Out-of-Sample 2022-2023)')
plt.xlabel('Date')
plt.ylabel('Net Asset Value (NAV)')
plt.legend()
plt.grid(True)
plt.show()