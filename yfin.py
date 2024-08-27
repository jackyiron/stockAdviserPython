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

    # 使用 history 方法獲取歷史數據
    historical_data = yf.download(
        tickers=f"{stock_no}.TW",
        period="max",
        interval="1mo",
        start=None,
        end=None,
        actions=True,
        auto_adjust=True,
        back_adjust=False
    )

    # 定義 CSV 文件的保存路徑
    output_path = os.path.join(output_dir, f"{stock_no}.csv")

    # 將數據保存為 CSV 文件
    historical_data.to_csv(output_path, encoding='utf-8')

    # 隨機等待 0.5 到 2 秒
    wait_time = random.uniform(0.5, 2.0)
    time.sleep(wait_time)
    print(f'{stock_no} done')
print("所有股票數據已保存至 stockData/month 目錄中的 CSV 文件。")

