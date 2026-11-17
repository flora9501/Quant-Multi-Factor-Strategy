# A-Share Cross-Sectional Momentum Strategy (截面动量反转策略)

## 📌 Project Overview (项目简介)
This project is an independent quantitative research pipeline that implements a cross-sectional momentum/reversal strategy on a pool of 100 Chinese A-share stocks. 

本项目是一个独立的量化研究工程，基于 A 股约 100 只主流股票的日频数据（2018-2023），构建截面动量反转策略。

## 🛠️ Methodology & Pipeline (核心方法论)
1. **Data Fetching**: Built a robust automated pipeline using `Tushare` API to fetch and store daily OHLCV data.
2. **Factor Construction**: Calculated 5-day momentum factor (`mom_5`).
3. **Crucial Data Preprocessing**: 
   - **Winsorization (去极值)**: Handled extreme outliers (1% and 99% percentiles) cross-sectionally.
   - **Z-score Standardization (标准化)**: Normalized factors daily.
   - **Tradability Filter (涨跌停过滤)**: Strictly excluded stocks hitting the daily price limit (>9.5% or <-9.5%) to prevent execution bias.
4. **Backtesting Engine**: 
   - Long the bottom 10 stocks and short the top 10 stocks daily.
   - **Look-ahead Bias Prevention**: Shifted target returns by `-1` day to strictly use T+1 returns.

## 📈 Results (回测业绩)
*Data Range: 2018 - 2023 (Contains 2021 market crash stress test)*
- **Annualized Return (年化收益)**: 16.19%
- **Annualized Volatility (年化波动)**: 27.72%
- **Sharpe Ratio (夏普比率)**: 0.58
- **Max Drawdown (最大回撤)**: -38.97%

*(Attach your green NAV chart here!)* 
![Strategy NAV](把你的净值图拖拽到 Github 网页里，它会自动生成一个链接，贴在这里)

## 🔍 Future Improvements (反思与优化方向)
*This is a Baseline strategy. The -38% drawdown in 2021 indicates a severe exposure to systematic risk (market/size factors).*
1. **Sector & Market Cap Neutralization (行业与市值中性化)**: Regress factors against market cap to remove size bias.
2. **Machine Learning**: Replace single-factor ranking with a `LightGBM` multi-factor model (incorporating volatility, volume, and momentum).
3. **Transaction Costs (交易成本)**: Add a realistic 0.2% transaction cost deduction into the backtester.

## 💻 Tech Stack
- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Matplotlib, SciPy, Tushare