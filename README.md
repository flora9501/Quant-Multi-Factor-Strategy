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



<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/ea952f7c-ffd1-4fad-8520-e7afe104d207" />

**Risk Analysis (风险归因)**:
*   **Severe Drawdown**: The significant drawdown of **-38.97%** during 2021 indicates a severe factor exposure to size (small-cap) and style (growth) risks.
*   **Root Cause**: Due to the lack of sector/market-cap neutralization, the strategy suffered heavily during the 2021 "core assets" crash.

<img width="1106" height="631" alt="截屏2026-11-17 23 18 08" src="https://github.com/user-attachments/assets/1bd5f25a-29f2-4b90-b2f3-db85eaa1bd3e" />

### V2.0 LightGBM Multi-Factor Model (Failed Baseline)

**Performance**: Annualized Return: -16.04%, Sharpe: -0.75 (2022-2023 Out-of-Sample).

**Research Note / Post-Mortem (复盘分析)**:
*   **Low Signal-to-Noise Ratio**: The model severely overfitted the training set (2018-2021). Tree-based models capture noise easily in small datasets.
*   **Distribution Shift**: The market regime shifted drastically in 2022. The model learned outdated non-linear relationships.

**Next Steps**:
1.  Expanding the stock pool to the entire A-share market to increase sample size.
2.  Implementing Market-Cap and Sector Neutralization.
3.  Switching to a Linear Model with L1 Regularization (Lasso) which is much more robust for time-series factor data.
