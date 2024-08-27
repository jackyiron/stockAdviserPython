import yfinance as yf
import pandas as pd
import time
import random
import os

# 以 UTF-8 編碼讀取 stockList.txt 第一欄的股票代號
with open('stockList.txt', 'r', encoding='utf-8') as f:
    stock_list = [line.split()[0] for line in f]

# 檢查並創建保存數據的目錄
output_dir = 'stockData/month'
os.makedirs(output_dir, exist_ok=True)

# 迴圈遍歷每個股票代號並獲取歷史數據
for stock_no in stock_list:
    # 創建股票對象
    data = yf.Ticker(f"{stock_no}.TW")
    # 使用 history 方法獲取歷史數據

    hist = data.earnings_estimate
    print(hist)
    exit()
    historical_data = data.history(
        tickers=f"{stock_no}.TW",
        period="1mo",
    )

    print(historical_data)


    # 隨機等待 0.5 到 2 秒
    wait_time = random.uniform(0.5, 2.0)
    time.sleep(wait_time)
    print(f'{stock_no} done')
print("所有股票數據已保存至 stockData/month 目錄中的 CSV 文件。")

