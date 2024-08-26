import pandas as pd
import json
import numpy as np
import random
import pandas as pd

def generate_mock_data():
    start_date = pd.to_datetime("2024-01-01")
    date_range = pd.date_range(start_date, periods=12, freq='M')  # 12 months of data

    mock_data = []
    for date in date_range:
        high = round(random.uniform(100, 200), 2)
        low = round(random.uniform(80, high), 2)
        average = round((high + low) / 2, 2)
        volume = random.randint(1000, 5000)

        mock_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "volume": volume,
            "high": high,
            "low": low,
            "average": average
        })

    return {"data": mock_data}

# 生成模擬數據
mock_json_data = generate_mock_data()

# 將模擬數據保存為 JSON 文件
with open('test_data.json', 'w') as f:
    json.dump(mock_json_data, f, indent=4)

# 輸出生成的數據
print(json.dumps(mock_json_data, indent=4))

def load_data_from_json(json_file):
    """
    從 JSON 檔案中讀取數據並轉換為 DataFrame
    :param json_file: JSON 檔案的路徑
    :return: 包含 volume, high, low, average 的 DataFrame
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 將 data 轉換為 DataFrame
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])  # 將日期轉換為 datetime 格式
    df.set_index('date', inplace=True)  # 設置日期為索引
    print(df) 
    return df
def calculate_vr(df, n=3):
    """
    計算 Volume Ratio (VR)
    :param df: 包含 Volume, High, Low, Average 欄位的 DataFrame
    :param n: 計算 VR 的期間
    :return: 包含 VR 值的 DataFrame
    """
    df['Previous Average'] = df['average'].shift(1)

    # 計算正、平、負成交量
    df['Positive Volume'] = df['volume'] * (df['average'] > df['Previous Average'])
    df['Neutral Volume'] = df['volume'] * (df['average'] == df['Previous Average']) * 0.5
    df['Negative Volume'] = df['volume'] * (df['average'] < df['Previous Average'])

    # 計算累積成交量
    df['Sum Positive Volume'] = df['Positive Volume'].rolling(window=n).sum()
    df['Sum Neutral Volume'] = df['Neutral Volume'].rolling(window=n).sum()
    df['Sum Negative Volume'] = df['Negative Volume'].rolling(window=n).sum()

    # 避免分母為 0 導致 inf
    min_non_zero_neg_vol = df['Sum Negative Volume'][df['Sum Negative Volume'] > 0].min()
    df['VR'] = (df['Sum Positive Volume'] + df['Sum Neutral Volume']) / np.where(df['Sum Negative Volume'] == 0, min_non_zero_neg_vol, df['Sum Negative Volume']) * 100

    return df[['VR']]

# 載入 JSON 數據
json_file = 'test_data.json'  # 替換為你的 JSON 檔案路徑
df = load_data_from_json(json_file)

# 計算 VR
vr_df = calculate_vr(df)
print(vr_df)  # 顯示最近計算出的 VR
