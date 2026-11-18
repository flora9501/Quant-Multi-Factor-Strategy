import akshare as ak
import pandas as pd
import time
import os

if not os.path.exists('data_ak'):
    os.makedirs('data_ak')

print("正在获取全 A 股股票列表...")
# 获取 A 股实时行情，顺便拿到所有股票代码
stock_spot = ak.stock_zh_a_spot_em()
# 我们取市值最大的前 200 只股票作为我们的股票池
stock_spot['总市值'] = pd.to_numeric(stock_spot['总市值'], errors='coerce')
stock_spot = stock_spot.sort_values('总市值', ascending=False).head(200)
stock_codes = stock_spot['代码'].tolist()

print(f"目标：拉取 {len(stock_codes)} 只股票（市值前200）的数据...")

for index, code in enumerate(stock_codes):
    file_name = f"data_ak/{code}.csv"
    if os.path.exists(file_name):
        continue
        
    print(f"[{index+1}/{len(stock_codes)}] 正在拉取: {code}")
    
    try:
        # 1. 拉取前复权日线行情 (注意接口名改成了 _em)
        df_price = ak.stock_zh_a_hist_em(
            symbol=code, period="daily", 
            start_date="20180101", end_date="20231231", adjust="qfq"
        )
        
        if df_price.empty:
            continue
            
        # 映射中文列名到英文
        df_price.rename(columns={
            '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'vol', 
            '涨跌幅': 'pct_chg', '换手率': 'turnover'
        }, inplace=True)
        
        # 2. 获取当前总市值 (用于后续中性化)
        try:
            info = ak.stock_individual_info_em(symbol=code)
            total_mv = info.loc[info['item'] == '总市值', 'value'].values[0]
            df_price['total_mv'] = pd.to_numeric(total_mv, errors='coerce')
        except:
            df_price['total_mv'] = float('nan')
            
        df_price['ts_code'] = code
        df_price.to_csv(file_name, index=False)
        
        time.sleep(0.3) # 稍微休眠，防封
        
    except Exception as e:
        print(f"拉取 {code} 失败: {e}")
        time.sleep(1)

print("数据拉取全部完成！请查看 data_ak 文件夹。")