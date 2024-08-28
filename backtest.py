import numpy as np
import pandas as pd

def backtest_strategy(estimated_price, interpolated_price, initial_cash=10000, unit_value=1000):
    """根据估计价格和插值价格进行回测并计算最终收益率"""
    
    # 将数据转化为 DataFrame
    df = pd.DataFrame({
        'estimated_price': estimated_price,
        'interpolated_price': interpolated_price
    })
    
    # 初始化变量
    cash = initial_cash
    position = 0  # 持仓数量
    last_action = None
    
    # 交易记录
    trades = []

    for i in range(1, len(df)):
        if df['estimated_price'].iloc[i] > df['interpolated_price'].iloc[i] and \
           df['estimated_price'].iloc[i-1] <= df['interpolated_price'].iloc[i-1]:
            # 上穿买入信号
            if cash >= unit_value:
                units_to_buy = cash // unit_value
                position += units_to_buy
                cash -= units_to_buy * unit_value
                trades.append((i, 'buy', units_to_buy, df['estimated_price'].iloc[i]))
                last_action = 'buy'
                
        elif df['estimated_price'].iloc[i] < df['interpolated_price'].iloc[i] and \
             df['estimated_price'].iloc[i-1] >= df['interpolated_price'].iloc[i-1]:
            # 下穿卖出信号
            if position > 0:
                units_to_sell = min(position, unit_value // df['estimated_price'].iloc[i])
                position -= units_to_sell
                cash += units_to_sell * df['estimated_price'].iloc[i]
                trades.append((i, 'sell', units_to_sell, df['estimated_price'].iloc[i]))
                last_action = 'sell'
                
    # 计算最终的资产总值
    final_value = cash + position * df['estimated_price'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    return {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'trades': trades
    }

# 示例数据
estimated_price = np.array([100, 102, 101, 104, 103, 105, 107])
interpolated_price = np.array([100, 101, 102, 103, 104, 106, 108])

# 进行回测
results = backtest_strategy(estimated_price, interpolated_price)
print("Initial Cash:", results['initial_cash'])
print("Final Value:", results['final_value'])
print("Total Return (%):", results['total_return'])
print("Trades:", results['trades'])

