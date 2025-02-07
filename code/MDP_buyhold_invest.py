import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import yfinance as yf

def getdatabyday(tiklst, st, ed):
    download_data = pd.DataFrame([])
    interval = '1d'
    LLL = len(tiklst)
    KKK = 1
    
    for tk in tiklst:
        stock = yf.Ticker(tk)
        stock_data = \
        stock.history(start = st, end = ed, interval= '1d').reset_index()
        if len(stock_data) > 0:
            stock_data['ticker'] = tk
            download_data = pd.concat([download_data, stock_data])
            print ('daily data tickname: ', tk)
            KKK += 1
        
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [1, 0])
    z = download_data['Date'].astype(str)
    download_data['Date'] = z
    dya_c = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    download_data = download_data[dya_c]
    tm_frame = pd.DataFrame(list(set(download_data['Date'])), columns = ['Date'])
    tm_frame = tm_frame.sort_values(['Date'], ascending = False)
    tm_frame['dayseq'] = range(1, len(tm_frame) + 1)
    download_data = pd.merge(download_data, tm_frame, on= ['Date'] , how='inner')
    download_data = download_data.sort_values(['ticker', 'Date'], ascending = [False, False])
    return download_data

tks = [       'jpm', 'ma', 'crm', , 'qcom', 'wmt', 'QQQ', 'SPY']

day_data = getdatabyday(tks, '2019-10-01', '2024-12-31')

day_data.shape
'''
(34346, 8)
'''

day_data.dtypes
'''
ticker     object
Date       object
Open      float64
High      float64
Low       float64
Close     float64
Volume      int64
dayseq      int64
'''

day_data.isnull().sum()
'''
ticker    0
Date      0
Open      0
High      0
Low       0
Close     0
Volume    0
dayseq    0
dtype: int64
'''

list(day_data.columns)
'''
['ticker', 'Date', 'Open', 'High',
 'Low', 'Close', 'Volume', 'dayseq']
'''
day_data.ticker.value_counts()

'''
zs      1321
wmt     1321
SPY     1321
aapl    1321
abbv    1321
acn     1321
amzn    1321
anet    1321
asml    1321
crm     1321
crwd    1321
goog    1321
isrg    1321
jpm     1321
lly     1321
ma      1321
meta    1321
mrvl    1321
msft    1321
now     1321
orcl    1321
panw    1321
qcom    1321
tmo     1321
tsm     1321
QQQ     1321
'''
day_data.Date.min()

'''
'2019-10-01 00:00:00-04:00'
'''

day_data.Date.max()
'''
'2024-12-30 00:00:00-05:00'
'''


import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# ------------------------------------------------------------------------------
# 1. Download Real Data from Yahoo Finance
# ------------------------------------------------------------------------------

def getdatabyday(tiklst, st, ed):
    download_data = pd.DataFrame([])
    for tk in tiklst:
        stock = yf.Ticker(tk)
        stock_data = stock.history(start=st, end=ed, interval='1d').reset_index()
        if len(stock_data) > 0:
            stock_data['ticker'] = tk
            download_data = pd.concat([download_data, stock_data])
            print('Downloaded daily data for ticker:', tk)
    download_data = download_data.sort_values(['ticker', 'Date'], ascending=[True, True])
    download_data['Date'] = download_data['Date'].astype(str)
    cols = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    download_data = download_data[cols]
    return download_data

# List of tickers including individual stocks and ETFs.
tks = ['goog', 'meta', 'msft', 'amzn', 'aapl', 'zs', 'panw', 'isrg',
       'jpm', 'ma', 'crm', 'asml','crwd', 'tsm', 'now', 'lly', 'acn',
       'tmo', 'abbv','anet','orcl', 'mrvl', 'qcom', 'wmt', 'QQQ', 'SPY']

# Download data (example period: 2019-10-01 to 2024-12-31)
day_data = getdatabyday(tks, '2019-10-01', '2024-12-31')
print("Downloaded data shape:", day_data.shape)
print("Data dtypes:")
print(day_data.dtypes)
print("Missing values:")
print(day_data.isnull().sum())
print("Ticker counts:")
print(day_data['ticker'].value_counts())

# ------------------------------------------------------------------------------
# 2. Hazard Computation (Simple Rolling Drawdown Risk)
# ------------------------------------------------------------------------------

def compute_hazard_signal(df: pd.DataFrame, window: int = 30, drawdown_thresh: float = 0.05) -> pd.DataFrame:
    """
    Computes a hazard-like signal for each stock on each date.
    An event is flagged if today's Close is at least drawdown_thresh (e.g. 5%) 
    below the rolling maximum over the past 'window' days.
    
    The hazard signal is then defined as the rolling average (over 'window' days) 
    of these event indicators.
    """
    # Ensure the ticker column is named "Ticker"
    if 'ticker' in df.columns and 'Ticker' not in df.columns:
        df = df.rename(columns={'ticker': 'Ticker'})
    df_sorted = df.sort_values(by=['Ticker', 'Date']).copy()
    df_sorted['rolling_max'] = (df_sorted.groupby('Ticker')['Close']
                                .transform(lambda s: s.rolling(window, min_periods=1).max()))
    df_sorted['drawdown_event'] = (((df_sorted['rolling_max'] - df_sorted['Close']) / df_sorted['rolling_max']) >= drawdown_thresh).astype(int)
    df_sorted['hazard_signal'] = (df_sorted.groupby('Ticker')['drawdown_event']
                                 .transform(lambda s: s.rolling(window, min_periods=1).mean()))
    df_sorted.drop(columns=['rolling_max', 'drawdown_event'], inplace=True)
    return df_sorted

# ------------------------------------------------------------------------------
# 3. Utility: Compute months difference between two dates
# ------------------------------------------------------------------------------

def months_diff(d1: datetime, d2: datetime) -> int:
    """
    Returns the approximate difference in full months between d1 and d2.
    Assumes d2 >= d1.
    """
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

# ------------------------------------------------------------------------------
# 4. MDP-Inspired Allocation with Balancing Mechanisms and Dynamic Fraction
# ------------------------------------------------------------------------------

def mdp_hazard_backtest(df_with_hazard: pd.DataFrame,
                        start_date: str,
                        end_date: str,
                        initial_capital: float = 100000.0,
                        hazard_threshold: float = 0.2,
                        max_daily_invest_fraction: float = 0.1,
                        max_invest_months: int = 18,
                        max_allocation_fraction: float = 0.2,  # Max allocation per stock as fraction of initial capital
                        cooldown_period: int = 5              # Cooldown period (days) before re-buying the same stock
                       ) -> pd.DataFrame:
    """
    Implements a dynamic investment strategy based on hazard signals.
    In each trading day:
      - If a stock's hazard signal is below a threshold and (a) its invested amount is under its cap and 
        (b) sufficient time has passed since its last purchase (cooldown),
        then invest a fraction of the remaining capital.
      - The fraction is dynamic: if hazard is much lower than the threshold, invest more.
      - After a specified number of months, any remaining capital is forcefully allocated.
    """
    # Filter data for the test period and sort by Date then Ticker (ascending order)
    mask = (df_with_hazard['Date'] >= start_date) & (df_with_hazard['Date'] <= end_date)
    df_period = df_with_hazard.loc[mask].copy()
    df_period.sort_values(by=['Date', 'Ticker'], inplace=True)
    
    # Dictionaries to track holdings and investment details per stock
    owned_shares = {}      # {Ticker: total number of shares owned}
    invested_amount = {}   # {Ticker: total capital invested in the stock}
    last_purchase = {}     # {Ticker: last purchase date as datetime}
    
    capital_remaining = initial_capital
    max_allocation = max_allocation_fraction * initial_capital  # Maximum allowed per stock
    
    daily_records = []
    start_dt = pd.to_datetime(start_date)
    
    unique_dates = df_period['Date'].sort_values().unique()
    
    for d in unique_dates:
        current_dt = pd.to_datetime(d)
        elapsed_months = months_diff(start_dt, current_dt)
        
        # Forced full investment if test period exceeds max_invest_months
        if (elapsed_months >= max_invest_months) and (capital_remaining > 0):
            day_df = df_period[df_period['Date'] == d]
            uninvested = day_df[~day_df['Ticker'].apply(lambda t: (invested_amount.get(t, 0) >= max_allocation))]
            if not uninvested.empty:
                n_candidates = len(uninvested)
                invest_per_stock = capital_remaining / n_candidates
                for _, row in uninvested.iterrows():
                    ticker = row['Ticker']
                    price = row['Close']
                    if price <= 0:
                        continue
                    allowed_investment = max_allocation - invested_amount.get(ticker, 0)
                    invest_amount = min(invest_per_stock, allowed_investment, capital_remaining)
                    num_shares = np.floor(invest_amount / price)
                    cost = num_shares * price
                    if num_shares > 0 and cost <= capital_remaining:
                        capital_remaining -= cost
                        owned_shares[ticker] = owned_shares.get(ticker, 0) + num_shares
                        invested_amount[ticker] = invested_amount.get(ticker, 0) + cost
                        last_purchase[ticker] = current_dt
                        print(f"[Forced] {d}: Bought {num_shares} shares of {ticker} at {price:.2f} for {cost:.2f}")
                        
        else:
            # Normal dynamic allocation: iterate over today's data
            day_df = df_period[df_period['Date'] == d]
            for _, row in day_df.iterrows():
                ticker = row['Ticker']
                price = row['Close']
                hazard = row['hazard_signal']
                if price <= 0:
                    continue
                if hazard < hazard_threshold:
                    # Enforce cooldown: if stock was purchased before, wait until cooldown period passes.
                    if ticker in last_purchase:
                        days_since_last = (current_dt - last_purchase[ticker]).days
                        if days_since_last < cooldown_period:
                            continue
                    # Skip if already fully allocated.
                    if invested_amount.get(ticker, 0) >= max_allocation:
                        continue
                    # Calculate dynamic fraction:
                    dynamic_fraction = max_daily_invest_fraction * (1 + (hazard_threshold - hazard) / hazard_threshold)
                    dynamic_fraction = min(dynamic_fraction, 2 * max_daily_invest_fraction)
                    investable = min(capital_remaining * dynamic_fraction,
                                     max_allocation - invested_amount.get(ticker, 0),
                                     capital_remaining)
                    num_shares = np.floor(investable / price)
                    cost = num_shares * price
                    if num_shares > 0 and cost <= capital_remaining:
                        capital_remaining -= cost
                        owned_shares[ticker] = owned_shares.get(ticker, 0) + num_shares
                        invested_amount[ticker] = invested_amount.get(ticker, 0) + cost
                        last_purchase[ticker] = current_dt
                        print(f"{d}: Bought {num_shares} shares of {ticker} at {price:.2f} for {cost:.2f}; Rem. capital: {capital_remaining:.2f}")
        
        # Compute portfolio value for day d: remaining capital + current market value of holdings
        day_value = capital_remaining
        day_df = df_period[df_period['Date'] == d]
        for tck, shares in owned_shares.items():
            ticker_rows = day_df[day_df['Ticker'] == tck]
            if not ticker_rows.empty:
                price = ticker_rows.iloc[0]['Close']
                day_value += shares * price
        daily_records.append({
            'Date': d,
            'PortfolioValue': day_value,
            'CapitalRemaining': capital_remaining,
            'NumStocksOwned': len(owned_shares)
        })
    
    return pd.DataFrame(daily_records)

# ------------------------------------------------------------------------------
# 5. Baseline Buy-and-Hold for QQQ and SPY
# ------------------------------------------------------------------------------

def baseline_buy_and_hold(benchmark_df: pd.DataFrame,
                          start_date: str,
                          end_date: str,
                          initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Implements a lump-sum buy-and-hold strategy for a benchmark ETF (QQQ or SPY)
    from start_date to end_date.
    """
    mask = (benchmark_df['Date'] >= start_date) & (benchmark_df['Date'] <= end_date)
    period_df = benchmark_df.loc[mask].sort_values(by='Date').copy()
    if period_df.empty:
        raise ValueError("Benchmark data is empty for the chosen period.")
    first_close = period_df.iloc[0]['Close']
    shares = np.floor(initial_capital / first_close)
    leftover = initial_capital - (shares * first_close)
    records = []
    for _, row in period_df.iterrows():
        px = row['Close']
        total_value = shares * px + leftover
        records.append({
            'Date': row['Date'],
            'Value': total_value
        })
    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 6. Full Backtest: Compare Dynamic Strategy vs. QQQ and SPY Buy-and-Hold
# ------------------------------------------------------------------------------

def run_full_backtest(DF: pd.DataFrame,
                      benchmark_df_QQQ: pd.DataFrame,
                      benchmark_df_SPY: pd.DataFrame,
                      test_start_date: str = "2020-01-01",
                      test_end_date: str = "2024-01-01",
                      initial_capital: float = 100000.0,
                      max_invest_months: int = 18,
                      max_allocation_fraction: float = 0.2,
                      cooldown_period: int = 5) -> pd.DataFrame:
    print("Starting full backtest.")
    print(f"Test period: {test_start_date} to {test_end_date}")
    print(f"Initial capital: {initial_capital}")
    
    # Compute hazard signals for the universe of stocks (non-ETF)
    df_hazard = compute_hazard_signal(DF, window=30, drawdown_thresh=0.05)
    print("Hazard signals computed.")
    
    hazard_threshold = 0.2  # Buy if hazard signal is below this threshold.
    mdp_results = mdp_hazard_backtest(
        df_with_hazard = df_hazard,
        start_date = test_start_date,
        end_date = test_end_date,
        initial_capital = initial_capital,
        hazard_threshold = hazard_threshold,
        max_daily_invest_fraction = 0.1,
        max_invest_months = max_invest_months,
        max_allocation_fraction = max_allocation_fraction,
        cooldown_period = cooldown_period
    )
    mdp_final_value = mdp_results.iloc[-1]['PortfolioValue'] if not mdp_results.empty else 0.0
    
    # Baseline: QQQ Buy-and-Hold
    qqq_results = baseline_buy_and_hold(benchmark_df_QQQ, test_start_date, test_end_date, initial_capital)
    qqq_final_value = qqq_results.iloc[-1]['Value'] if not qqq_results.empty else 0.0
    
    # Baseline: SPY Buy-and-Hold
    spy_results = baseline_buy_and_hold(benchmark_df_SPY, test_start_date, test_end_date, initial_capital)
    spy_final_value = spy_results.iloc[-1]['Value'] if not spy_results.empty else 0.0
    
    # Calculate percentage improvement over benchmarks
    improvement_over_QQQ = ((mdp_final_value - qqq_final_value) / qqq_final_value) * 100
    improvement_over_SPY = ((mdp_final_value - spy_final_value) / spy_final_value) * 100
    
    print(f"MDP+Hazard final portfolio value: {mdp_final_value:,.2f}")
    print(f"QQQ Buy-and-Hold final portfolio value: {qqq_final_value:,.2f}")
    print(f"SPY Buy-and-Hold final portfolio value: {spy_final_value:,.2f}")
    print(f"Improvement over QQQ: {improvement_over_QQQ:.2f}%")
    print(f"Improvement over SPY: {improvement_over_SPY:.2f}%")
    
    # Merge benchmark series for plotting and comparison.
    comparison_df = pd.merge(mdp_results[['Date', 'PortfolioValue']],
                             qqq_results.rename(columns={'Value': 'QQQValue'})[['Date', 'QQQValue']],
                             on='Date', how='outer')
    comparison_df = pd.merge(comparison_df,
                             spy_results.rename(columns={'Value': 'SPYValue'})[['Date', 'SPYValue']],
                             on='Date', how='outer')
    comparison_df.sort_values('Date', inplace=True)
    
    return comparison_df

# ------------------------------------------------------------------------------
# 7. Example Usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Rename 'ticker' to 'Ticker' for consistency.
    DF_all = day_data.rename(columns={'ticker': 'Ticker'})
    
    # For the dynamic strategy, use non-ETF stocks (exclude QQQ and SPY).
    DF = DF_all[~DF_all['Ticker'].isin(['QQQ', 'SPY'])].copy()
    # Create baseline DataFrames for QQQ and SPY.
    QQQ_DF = DF_all[DF_all['Ticker'] == 'QQQ'].copy()
    SPY_DF = DF_all[DF_all['Ticker'] == 'SPY'].copy()
    
    print("Dynamic strategy (stocks) data shape:", DF.shape)
    print("QQQ data shape:", QQQ_DF.shape)
    print("SPY data shape:", SPY_DF.shape)
    
    # Define test period (4 years) for performance comparison.
    test_start_date = "2020-01-01"
    test_end_date = "2024-01-01"
    
    comparison_df = run_full_backtest(
        DF=DF,
        benchmark_df_QQQ=QQQ_DF,
        benchmark_df_SPY=SPY_DF,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        initial_capital=100000.0,
        max_invest_months=18,
        max_allocation_fraction=0.2,
        cooldown_period=5
    )
    
    print("Backtest complete. Final comparison data (last 5 rows):")
    print(comparison_df.tail())
    
    # Optional: Plot the results.
    # import matplotlib.pyplot as plt
    # comparison_df.plot(x='Date', y=['PortfolioValue', 'QQQValue', 'SPYValue'], figsize=(10, 6))
    # plt.title("Dynamic MDP+Hazard Strategy vs. QQQ and SPY Buy-and-Hold")
    # plt.xlabel("Date")
    # plt.ylabel("Portfolio Value")
    # plt.show()


'''
Name: count, dtype: int64
Dynamic strategy (stocks) data shape: (31704, 7)
QQQ data shape: (1321, 7)
SPY data shape: (1321, 7)
Starting full backtest.
Test period: 2020-01-01 to 2024-01-01
Initial capital: 100000.0
Hazard signals computed.
2020-01-02 00:00:00-05:00: Bought 274.0 shares of aapl at 72.80 for 19946.11; Rem. capital: 80053.89
2020-01-02 00:00:00-05:00: Bought 223.0 shares of abbv at 71.59 for 15964.52; Rem. capital: 64089.37
2020-01-02 00:00:00-05:00: Bought 65.0 shares of acn at 195.26 for 12692.13; Rem. capital: 51397.24
2020-01-02 00:00:00-05:00: Bought 108.0 shares of amzn at 94.90 for 10249.25; Rem. capital: 41147.98
2020-01-02 00:00:00-05:00: Bought 28.0 shares of asml at 289.48 for 8105.40; Rem. capital: 33042.58
2020-01-02 00:00:00-05:00: Bought 39.0 shares of crm at 166.06 for 6476.44; Rem. capital: 26566.14
2020-01-02 00:00:00-05:00: Bought 77.0 shares of goog at 68.12 for 5245.53; Rem. capital: 21320.62
2020-01-02 00:00:00-05:00: Bought 21.0 shares of isrg at 199.09 for 4180.82; Rem. capital: 17139.80
2020-01-02 00:00:00-05:00: Bought 28.0 shares of jpm at 121.48 for 3401.36; Rem. capital: 13738.43
2020-01-02 00:00:00-05:00: Bought 22.0 shares of lly at 123.92 for 2726.14; Rem. capital: 11012.29
2020-01-02 00:00:00-05:00: Bought 7.0 shares of ma at 294.59 for 2062.13; Rem. capital: 8950.17
2020-01-02 00:00:00-05:00: Bought 8.0 shares of meta at 208.98 for 1671.85; Rem. capital: 7278.31
2020-01-02 00:00:00-05:00: Bought 9.0 shares of msft at 153.63 for 1382.68; Rem. capital: 5895.64
2020-01-02 00:00:00-05:00: Bought 3.0 shares of now at 291.24 for 873.72; Rem. capital: 5021.92
2020-01-02 00:00:00-05:00: Bought 3.0 shares of tmo at 322.47 for 967.42; Rem. capital: 4054.49
2020-01-02 00:00:00-05:00: Bought 14.0 shares of tsm at 54.50 for 763.06; Rem. capital: 3291.43
2020-01-02 00:00:00-05:00: Bought 17.0 shares of wmt at 36.78 for 625.30; Rem. capital: 2666.13
2020-01-07 00:00:00-05:00: Bought 7.0 shares of abbv at 71.06 for 497.44; Rem. capital: 2168.69
2020-01-07 00:00:00-05:00: Bought 2.0 shares of acn at 189.48 for 378.97; Rem. capital: 1789.72
2020-01-07 00:00:00-05:00: Bought 3.0 shares of amzn at 95.34 for 286.03; Rem. capital: 1503.69
2020-01-07 00:00:00-05:00: Bought 1.0 shares of asml at 285.62 for 285.62; Rem. capital: 1218.08
2020-01-07 00:00:00-05:00: Bought 1.0 shares of crm at 175.02 for 175.02; Rem. capital: 1043.06
2020-01-07 00:00:00-05:00: Bought 3.0 shares of goog at 69.42 for 208.25; Rem. capital: 834.80
2020-01-07 00:00:00-05:00: Bought 1.0 shares of jpm at 117.74 for 117.74; Rem. capital: 717.06
2020-01-07 00:00:00-05:00: Bought 1.0 shares of lly at 124.20 for 124.20; Rem. capital: 592.86
2020-01-07 00:00:00-05:00: Bought 2.0 shares of tsm at 52.94 for 105.89; Rem. capital: 486.98
2020-01-07 00:00:00-05:00: Bought 2.0 shares of wmt at 36.05 for 72.09; Rem. capital: 414.88
2020-01-13 00:00:00-05:00: Bought 1.0 shares of abbv at 70.77 for 70.77; Rem. capital: 344.12
2020-01-13 00:00:00-05:00: Bought 1.0 shares of tsm at 54.52 for 54.52; Rem. capital: 289.60
2020-01-13 00:00:00-05:00: Bought 1.0 shares of wmt at 35.84 for 35.84; Rem. capital: 253.76
2020-01-21 00:00:00-05:00: Bought 2.0 shares of anet at 13.83 for 27.66; Rem. capital: 226.10
2020-01-21 00:00:00-05:00: Bought 1.0 shares of wmt at 35.75 for 35.75; Rem. capital: 190.36
2020-01-27 00:00:00-05:00: Bought 2.0 shares of anet at 14.69 for 29.37; Rem. capital: 160.99
2020-02-03 00:00:00-05:00: Bought 1.0 shares of anet at 14.08 for 14.08; Rem. capital: 146.91
2020-02-10 00:00:00-05:00: Bought 1.0 shares of anet at 14.53 for 14.53; Rem. capital: 132.38
2020-02-18 00:00:00-05:00: Bought 1.0 shares of anet at 14.11 for 14.11; Rem. capital: 118.27
2020-12-10 00:00:00-05:00: Bought 1.0 shares of anet at 17.27 for 17.27; Rem. capital: 101.00
2020-12-15 00:00:00-05:00: Bought 1.0 shares of anet at 17.47 for 17.47; Rem. capital: 83.52
MDP+Hazard final portfolio value: 211,686.77
QQQ Buy-and-Hold final portfolio value: 194,189.68
SPY Buy-and-Hold final portfolio value: 155,807.67
Improvement over QQQ: 9.01%
Improvement over SPY: 35.86%
Backtest complete. Final comparison data (last 5 rows):
                           Date  PortfolioValue       QQQValue       SPYValue
1001  2023-12-22 00:00:00-05:00   212046.460161  193547.308884  155263.521606
1002  2023-12-26 00:00:00-05:00   212151.369387  194732.166519  155919.102051
1003  2023-12-27 00:00:00-05:00   212531.373884  195128.579605  156201.030518
1004  2023-12-28 00:00:00-05:00   212296.304674  195033.726944  156260.028198
1005  2023-12-29 00:00:00-05:00   211686.769300  194189.675095  155807.672119

'''

