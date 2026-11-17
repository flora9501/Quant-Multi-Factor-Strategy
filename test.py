import tushare as ts
ts.set_token('066c36e0b8185533c45cbbcc911277566a2fd6852bc8bae4d363324e')
pro = ts.pro_api()

# 测试一只主板股票（应该能返回数据）
print(pro.daily(ts_code='600519.SH', start_date='20230101', end_date='20230110').shape)

# 测试一只 920 开头的北交所股票
print(pro.daily(ts_code='920288.BJ', start_date='20230101', end_date='20231231').shape)