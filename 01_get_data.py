import tushare as ts
import pandas as pd
import time
import os

# 1. 初始化
ts.set_token('请输入你的TOKEN')
pro = ts.pro_api()

# 2. 建文件夹
if not os.path.exists('data'):
    os.makedirs('data')

# 3. 直接写死 100 只具有代表性的股票代码（包含沪深主板、创业板等）
# 这样就能完美避开 stock_basic 接口的调用限制！
stock_codes = [
    '600519.SH', '601318.SH', '600036.SH', '601166.SH', '600900.SH',
    '600276.SH', '601888.SH', '600030.SH', '601012.SH', '600887.SH',
    '601398.SH', '601288.SH', '601988.SH', '601939.SH', '600028.SH',
    '601857.SH', '600050.SH', '601088.SH', '601668.SH', '601818.SH',
    '000858.SZ', '000333.SZ', '000651.SZ', '000001.SZ', '000002.SZ',
    '000063.SZ', '000100.SZ', '000725.SZ', '002415.SZ', '002594.SZ',
    '300750.SZ', '300059.SZ', '300015.SZ', '300124.SZ', '300014.SZ',
    '600309.SH', '600690.SH', '600104.SH', '601601.SH', '601628.SH',
    '601211.SH', '601336.SH', '600809.SH', '600438.SH', '601899.SH',
    '603259.SH', '603288.SH', '600031.SH', '600048.SH', '601633.SH',
    '600104.SH', '600585.SH', '600406.SH', '600588.SH', '600570.SH',
    '002230.SZ', '002475.SZ', '002714.SZ', '300142.SZ', '300274.SZ',
    '300122.SZ', '300347.SZ', '300408.SZ', '300450.SZ', '300496.SZ',
    '601919.SH', '601390.SH', '601186.SH', '601766.SH', '601989.SH',
    '600016.SH', '600000.SH', '601328.SH', '601658.SH', '601138.SH',
    '000568.SZ', '000538.SZ', '000876.SZ', '000895.SZ', '002304.SZ',
    '002027.SZ', '002352.SZ', '002493.SZ', '002812.SZ', '002821.SZ',
    '600183.SH', '600745.SH', '600893.SH', '600760.SH', '601689.SH',
    '603501.SH', '603986.SH', '688981.SH', '688036.SH', '688012.SH'
]

print(f"目标：拉取 {len(stock_codes)} 只股票从 2018 到 2023 的数据。")

# 4. 循环拉取并保存
for index, ts_code in enumerate(stock_codes):
    file_name = f"data/{ts_code}.csv"
    
    if os.path.exists(file_name):
        print(f"[{index+1}/{len(stock_codes)}] {ts_code} 已存在，跳过。")
        continue
        
    print(f"[{index+1}/{len(stock_codes)}] 正在拉取: {ts_code}")
    
    try:
        df = pro.daily(ts_code=ts_code, start_date='20180101', end_date='20231231')
        
        if not df.empty:
            df.to_csv(file_name, index=False)
        else:
            print(f"  警告: {ts_code} 数据为空")
            
        # 免费接口有频次限制，每次请求后必须休眠
        time.sleep(1.5) 
        
    except Exception as e:
        print(f"  报错: {ts_code} 拉取失败。错误信息: {e}")
        time.sleep(3) 

print("\n数据拉取全部完成！请查看 data 文件夹。")