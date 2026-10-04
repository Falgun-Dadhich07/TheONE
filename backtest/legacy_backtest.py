# LEGACY: the team's original single-asset backtester, kept verbatim for the record.
# DO NOT USE. Superseded by backtest/engine.py. Audit findings (see docs/strategy_draft.md):
#  1. Lookahead: entries are gated on no_of_sell_signals / no_of_buy_signals,
#     which count signals over the WHOLE dataset, including the future.
#  2. Fills at close[i] of the same bar that produced the signal (should be next bar's open).
#  3. No fees, spread or slippage.
#  4. A repeated signal in the same direction overwrites entry_price/shares
#     without booking the open P&L.
#  5. The ATR stop uses atr[i] of the current bar (a moving stop), checks only
#     the close (not high/low), and fills at that close.
#  6. Sizing uses absolute ATR thresholds (2000/4000) that only make sense for
#     BTC; int() share rounding drops fractional coins.
#  7. Shorts sized at full capital with no margin model (shorts are off for us).
#  8. Sharpe/Sortino commented out, no annualization, no Calmar; returns in %;
#     "days" label on holding time that is actually counted in bars;
#     division by zero when there are no trades.
#  9. Single asset only.
# Correct parts kept in the new code: max/average drawdown, benchmark return.
#
# Note: as pasted, this file has no imports (it expects `df` and `plt` to exist).

def backtest(df):
  opening_capital = 1000000
  capt = opening_capital
  signals = list(df['signals'])
  close = list(df['Close'])
  no_of_trades = 0
  long_trades = 0
  winning_trades = 0
  atr = list(df['atr'])
  current_trade = 0
  entry_price = 0
  shares = 0
  no_of_buy_signals = len(df[df['signals'] == 1])
  no_of_sell_signals = len(df[df['signals'] == -1])
  portfolio_value = []
  ht = []
  entry_index = 0
  for i in range(len(df)):
    if current_trade:
      if current_trade == 1:
        if close[i] < (entry_price - atr[i]):
          capt+=(shares*(close[i] - entry_price))
          current_trade = 0
          shares = 0
          no_of_trades+=1
          long_trades+=1
          ht.append(i - entry_index)
        if current_trade:
          portfolio_value.append(capt+(shares*(close[i] - entry_price)))
        else:
          portfolio_value.append(capt)
      else:
        if close[i] > (entry_price + atr[i]):
          capt+=(shares*(entry_price - close[i]))
          current_trade = 0
          shares = 0
          no_of_trades+=1
          ht.append(i - entry_index)
        if current_trade:
          portfolio_value.append(capt+(shares*(entry_price - close[i])))
        else:
          portfolio_value.append(capt)
    else:
      portfolio_value.append(capt)
    if signals[i] == 1:
      if current_trade == -1:
        capt+=(shares*(entry_price - close[i]))
        if entry_price > close[i]:
          winning_trades+=1
        current_trade = 0
        shares = 0
        no_of_trades+=1
        ht.append(i - entry_index)
      if no_of_sell_signals > 0:
        current_trade = 1
        if atr[i] <= 2000:
          shares = int(capt/close[i])
        elif atr[i] > 2000 and atr[i] < 4000:
          shares = int(0.75*capt/close[i])
        else:
          shares = int(0.5*capt/close[i])
        entry_price = close[i]
        entry_index = i
      no_of_buy_signals-=1
    if signals[i] == -1:
      if current_trade == 1:
        capt+=(shares*(close[i] - entry_price))
        current_trade = 0
        shares = 0
        no_of_trades+=1
        long_trades+=1
        ht.append(i - entry_index)
        if close[i] > entry_price:
          winning_trades+=1
      if no_of_buy_signals > 0:
        current_trade = -1
        entry_price = close[i]
        if atr[i] <= 2000:
          shares = int(capt/close[i])
        elif atr[i] > 2000 and atr[i] < 4000:
          shares = int(0.75*capt/close[i])
        else:
          shares = int(0.5*capt/close[i])
        entry_index = i
      no_of_sell_signals-=1
  df['portfolio_value'] = portfolio_value
  dd = []
  peak = portfolio_value[0]
  for value in portfolio_value:
      if value > peak:
          peak = value
      drawdown = (peak - value) / peak
      dd.append(drawdown)

  max_drawdown = max(dd)
  average_drawdown = sum(dd) / len(dd)
  excess_returns = [0]
  for i in range(1,len(portfolio_value)):
    er = (portfolio_value[i] - portfolio_value[i-1])*100/portfolio_value[i-1]
    excess_returns.append(er)
  df['excess_returns'] = excess_returns
  # sr1 = sharpe_ratio(df['excess_returns'],len(df),0.04)
  # sr2 = sortino_ratio(df['excess_returns'],len(df),0.04)
  print(f'Benchmark Returns: {(close[len(df)-1] - close[0])*100/close[0]}%')
  print(f'Portfolio Returns: {(portfolio_value[len(df)-1] - portfolio_value[0])*100/portfolio_value[0]}%')
  print(f'Total trades executed: {no_of_trades}')
  print(f'Win rate: {winning_trades*100/no_of_trades}%')
  print(f'Total Long Trades: {long_trades}')
  print(f'Total Short Trades: {no_of_trades - long_trades}')
  print(f'Average Drawdown: -{average_drawdown*100}%')
  print(f'Max Drawdown: -{max_drawdown*100}%')
  # print(f'Sharpe Ratio: {sr1}')
  # print(f'Sortino Ratio: {sr2}')
  print(f'Average Holding Time: {sum(ht)/len(ht)} days')
  print(f'Maximum Holding Time: {max(ht)} days')
  print()
  plt.figure(figsize=(12,6))
  plt.plot(df['portfolio_value'])
  plt.title('Portfolio Value')
  plt.show()
# backtest(df)  # (call disabled so importing this file has no side effects)
