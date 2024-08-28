import yfinance as yf
import pandas as pd
import time
import random
import os
import requests_cache

from yahoo_fin.stock_info import *


from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pandas as pd
from twstock import Stock

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


# 以 UTF-8 編碼讀取 stockList.txt 第一欄的股票代號
with open('stockList3.txt', 'r', encoding='utf-8') as f:
    stock_list = [line.split()[0] for line in f]

# 檢查並創建保存數據的目錄
output_dir = 'stockData/month'
os.makedirs(output_dir, exist_ok=True)

# 定義檢查文件大小的閾值（3KB）
SIZE_THRESHOLD_KB = 3
SIZE_THRESHOLD_BYTES = SIZE_THRESHOLD_KB * 1024  # 3KB in bytes

# 迴圈遍歷每個股票代號並獲取歷史數據
for stock_no in stock_list:
    output_path = os.path.join(output_dir, f"{stock_no}.csv")

    try:
        # 檢查文件是否存在且大小是否小於閾值
        if  os.path.exists(output_path) and os.path.getsize(output_path) > SIZE_THRESHOLD_BYTES :
            print(f'{stock_no} 已存在且大於 3KB，跳過抓取')
            continue
    except:
        pass

    # 創建股票對象
    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND * 6)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )

    session.headers[
        'User-agent'] = f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'

    stock_no = '1101'
    stock = Stock(stock_no)
    data = stock.fetch(2024,1)

    df = pd.DataFrame(data)
    print(df.iloc[-2:])
    exit()

#    ticker = yf.Ticker(f'{stock_no}', session=session)

#    # The scraped response will be stored in the cache
#    historical_data = yf.download(
#        tickers=f"{stock_no}.TW",
#        period="max",
#        interval="1mo",
#        start=None,
#        end=None,
#        actions=True,
#        auto_adjust=True,
#        back_adjust=False
#    )
#
#    # 使用 history 方法獲取歷史數據
#    if not historical_data.empty:
#        # 將數據保存為 CSV 文件
#        historical_data.to_csv(output_path, encoding='utf-8')
#        print(f'{stock_no} 的數據已保存至 {output_path}')
#        # 隨機等待 3 到 8 秒
#        wait_time = random.uniform(3, 8.0)
#        time.sleep(wait_time)
#    else:
#        # 創建一個空的 CSV 文件
#        with open(output_path, 'w', encoding='utf-8') as f:
#            f.write('Date,Open,High,Low,Close,Adj Close,Volume\n')  # Create headers for an empty file
#        print(f'{stock_no} 的數據為空，已創建空文件 {output_path}')

        # 隨機等待 3 到 8 秒
#        wait_time = random.uniform(3, 8.0)
#        time.sleep(wait_time)



print("所有股票數據已保存至 stockData/month 目錄中的 CSV 文件。")
